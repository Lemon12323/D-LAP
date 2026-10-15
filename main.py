import os
import sys
import argparse
import time
import json
import csv
import hashlib
import platform
from importlib import metadata as package_metadata
from datetime import datetime

import torch
from torchvision import transforms
import numpy as np
import random
import copy
import gc
from PIL import Image

from config import exp_configuration
from utils import (
    load_ground_truth,
    WrapperModel,
    load_model,
    EvalResult,
    save_images,
    get_available_local_vit_models,
    LOCAL_VIT_MODEL_NAMES,
)
from attacks import ftm_attack
from dlap_attack import dlap_ftm_attack
from clean_repeatgrad import clean_repeatgrad_attack
from tsnr_attack import tsnr_ftm_attack
from temporal_decomp import temporal_decomp_probe_attack
from stateful_param_walk import stateful_param_walk_attack
from cross_space_reliability import cross_space_reliability_attack
from memory_utils import (CudaMemoryTracker, MemoryPolicy, generation_signature,
                          likely_memory_consumers, validate_preflight)

now = datetime.now()
today_string = now.strftime("%Y-%m-%d|%H:%M:%S")


def resolve_image_path(image_dir, image_identifier):
    """Resolve an image id to an on-disk file while tolerating common suffixes."""
    direct_path = os.path.join(image_dir, image_identifier)
    if os.path.isfile(direct_path):
        return direct_path

    stem = os.path.splitext(image_identifier)[0]
    for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
        candidate = os.path.join(image_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"Cannot find image file for identifier: {image_identifier}")


def format_hms(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def should_log_by_interval(processed, last_logged, interval, total):
    if processed >= total:
        return True
    return (processed - last_logged) >= interval


def build_adv_filename(img_idx, image_id, label_ori, label_tar):
    return f"{img_idx}_{image_id}_{label_ori}_{label_tar}.png"


def build_attack_timing_metrics(attack_elapsed_seconds, attacked_images, peak_cuda_memory_bytes=0):
    attacked_images = int(max(0, attacked_images))
    attack_elapsed_seconds = float(max(0.0, attack_elapsed_seconds))
    metrics = {
        "attack_elapsed_seconds": attack_elapsed_seconds,
        "attack_elapsed_hms": format_hms(attack_elapsed_seconds),
        "attacked_images": attacked_images,
        "attack_images_per_second": 0.0,
        "attack_milliseconds_per_image": 0.0,
        "peak_cuda_memory_bytes": int(peak_cuda_memory_bytes),
        "peak_cuda_memory_gib": float(peak_cuda_memory_bytes) / (1024 ** 3),
    }
    if attacked_images > 0 and attack_elapsed_seconds > 0:
        metrics["attack_images_per_second"] = attacked_images / attack_elapsed_seconds
        metrics["attack_milliseconds_per_image"] = (
            attack_elapsed_seconds * 1000.0 / attacked_images
        )
    return metrics


def save_attack_timing(save_dir, metrics):
    path = os.path.join(save_dir, "attack_timing.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


COMPUTE_KEYS = (
    "logical_attack_forward_calls",
    "source_model_forward_executions",
    "backward_calls",
    "image_gradient_evaluations",
    "parameter_gradient_evaluations",
    "parameter_state_refresh_steps",
    "parameter_state_nonrefresh_steps",
    "parameter_state_sample1_grads",
    "parameter_state_sample2_grads",
    "parameter_state_adaptive_updates",
    "parameter_state_freeze_checks",
    "parameter_restore_checks",
    "adversarial_image_updates",
    "clean_feature_cache_forwards",
    "sampled_parameter_neighborhoods",
    "parameter_perturbations_applied",
    "effective_attack_steps",
    "m1_image_update_steps",
    "m2_image_update_steps",
    "sample2_trigger_count",
    "ftm_final_sample_state_update_steps",
    "ftm_tuning_tensor_updates",
    "random_direction_draw_calls",
    "random_direction_tensor_draw_calls",
    "antithetic_even_fresh_directions",
    "antithetic_odd_cached_directions",
    "antithetic_rng_discarded_directions",
    "antithetic_completed_pairs",
    "antithetic_unpaired_final_steps",
    "rtmf_transform_calls",
    "rtmf_transform_random_scalar_draws",
    "ftm_mixing_decision_draw_calls",
    "ftm_shuffle_draw_calls",
    "ftm_coefficient_draw_calls",
    "guidance_refresh_count",
    "early_guidance_refresh_count",
    "late_guidance_refresh_count",
    "early_phase_image_updates",
    "late_phase_image_updates",
    "early_phase_parameter_perturbations",
    "late_phase_parameter_perturbations",
    "tsnr_reliability_steps",
    "tsnr_filtered_updates",
    "temporal_ema_updates",
    "tsnr_probe_checkpoints",
    "p06_decomp_steps",
    "p06_probe_checkpoints",
    "spw_steps",
    "spw_parameter_updates",
    "spw_projection_clips",
    "spw_restore_checks",
    "spw_selected_groups",
    "spw_skipped_normalization_tensors",
    "csr_steps",
    "csr_feature_only_steps",
    "csr_cross_space_steps",
    "csr_param_grad_requests",
    "csr_active_layers",
    "csr_weighted_model_steps",
    "csr_single_active_model_steps",
    "csr_weight_max_sum",
    "csr_weight_entropy_sum",
    "csr_score_std_sum",
    "csr_feature_sensitivity_sum",
    "csr_param_sensitivity_sum",
)


def environment_metadata(device):
    def version(name):
        try:
            return package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            return None

    has_cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": version("torchvision"),
        "timm": version("timm"),
        "cuda": torch.version.cuda if has_cuda else None,
        "cudnn": torch.backends.cudnn.version() if has_cuda else None,
        "gpu": torch.cuda.get_device_name(device) if has_cuda and str(device).startswith("cuda") else None,
    }


def check_new_run_directory(save_dir):
    protected = ("attack_params.json", "attack_timing.json", "compute_stats.json", "memory_stats.json",
                 "sample_manifest.csv", "eval_results.txt", "evaluation_model_weights.json",
                 "evaluation_config.json", "evaluation_memory_stats.json")
    for name in protected:
        if os.path.exists(os.path.join(save_dir, name)):
            raise FileExistsError(f"Refusing to overwrite existing run artifact: {os.path.join(save_dir, name)}")
    adv_dir = os.path.join(save_dir, "adv_imgs")
    if os.path.isdir(adv_dir) and os.listdir(adv_dir):
        raise FileExistsError(f"Refusing to overwrite existing adversarial images: {adv_dir}")


def validate_samples(records):
    if not records:
        raise ValueError("Dataset selection is empty")
    names = [r["filename"] for r in records]
    if len(names) != len(set(names)):
        raise ValueError("Dataset filenames must be unique")
    for r in records:
        for key in ("true_label", "target_label"):
            if not 0 <= int(r[key]) < 1000:
                raise ValueError(f"Invalid {key} for {r['filename']}: {r[key]}")
        if int(r["true_label"]) == int(r["target_label"]):
            raise ValueError(f"Target equals true label for {r['filename']}")


def save_sample_manifest(save_dir, records):
    path = os.path.join(save_dir, "sample_manifest.csv")
    columns = ("generation_index", "filename", "true_label", "target_label",
               "raw_true_label", "raw_target_label", "seed", "source_model")
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sample_manifest(path, generation_metadata):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"generation_index", "filename", "true_label", "target_label",
                    "seed", "source_model"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Incomplete generation manifest: {path}")
        rows = list(reader)
    if not rows or len(rows) != generation_metadata.get("num_images"):
        raise ValueError("Generation manifest image count differs from attack_params.json")
    source = generation_metadata.get("source_model_name")
    seed = generation_metadata.get("seed")
    for index, row in enumerate(rows):
        if int(row["generation_index"]) != index or row["source_model"] != source or int(row["seed"]) != seed:
            raise ValueError(f"Inconsistent generation manifest at row {index}")
    validate_samples(rows)
    return rows


def load_generation_manifest_input(path, expected_seed, source_model, expected_count):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"generation_index", "filename", "true_label", "target_label",
                "raw_true_label", "raw_target_label", "seed", "source_model"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Incomplete frozen generation manifest: {path}")
    if len(rows) != expected_count:
        raise ValueError(
            f"Frozen manifest has {len(rows)} rows; expected exactly {expected_count}"
        )
    for index, row in enumerate(rows):
        if int(row["generation_index"]) != index:
            raise ValueError(f"Non-contiguous generation_index at row {index}")
        if int(row["seed"]) != expected_seed:
            raise ValueError(f"Frozen manifest seed differs at row {index}")
        if row["source_model"] != source_model:
            raise ValueError(f"Frozen manifest source model differs at row {index}")
    validate_samples(rows)
    return rows


def save_compute_stats(save_dir, totals, batches):
    path = os.path.join(save_dir, "compute_stats.json")
    payload = {"counter_scope": "per batch; one logical gradient evaluation covers the entire image batch",
               "totals": totals, "batches": batches}
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return path


def build_attack_params(args, exp_settings, total_img_num, atk_type, source_model_name,
                        source_weight_info, manifest_path, label_info):
    """Persist the exact attack configuration for reproducibility and review."""
    num_iter = exp_settings.get("max_iterations", 300)
    epsilon = exp_settings.get("epsilon", 16)
    consumed_iteration = 1 if atk_type and "F" in atk_type else 0
    total_steps = max(num_iter - consumed_iteration, 0)
    interval_active = bool(
        args.use_dlap and args.lap_num_samples > 1 and args.lap_num_samples_interval > 1
    )

    return {
        "protocol_version": "p0a-1",
        "timestamp": today_string,
        "command": " ".join(sys.argv),
        "save_dir": args.save_dir,
        "mode": "generation",
        "attack_type": atk_type,
        "num_iter": num_iter,
        "consumed_iteration": consumed_iteration,
        "total_steps": total_steps,
        "num_images": total_img_num,
        "epsilon": epsilon,
        "alpha": exp_settings.get("alpha"),
        "seed": args.seed,
        "experiment_condition": args.experiment_condition,
        "source_model_name": source_model_name,
        "source_weights": source_weight_info,
        "environment": environment_metadata(args.device),
        "generation_batch_size": args.batch_size,
        "evaluation_batch_size": args.eval_batch_size or args.batch_size,
        "memory_policy": {
            "budget_gib": args.vram_budget_gb, "warn_gib": args.vram_warn_gb,
            "strict": args.vram_strict, "monitor": args.vram_monitor,
            "preflight": args.vram_preflight,
            "preflight_report": args.vram_preflight_report or None,
            "amp_enabled": False, "generation_auto_resize_enabled": False,
            "memory_stats_file": "memory_stats.json",
        },
        "dataset": {
            "image_csv": os.path.abspath(args.image_csv),
            "image_root": os.path.abspath(args.image_dir),
            "sample_manifest": os.path.abspath(manifest_path),
            "sample_manifest_sha256": file_sha256(manifest_path),
            "sample_manifest_input": (os.path.abspath(args.sample_manifest_input)
                                      if args.sample_manifest_input else None),
            "num_samples": total_img_num,
            "label_normalization": label_info,
        },
        "effective_image_step_size": exp_settings.get("alpha", 1.6) / 255.0,
        "effective_epsilon": epsilon / 255.0,
        "use_dlap": args.use_dlap,
        "attack_control_mode": args.dlap_control_mode if args.use_dlap else "standard",
        "p03_mode": getattr(args, "p03_mode", "none"),
        "p03_calibration_sha256": (file_sha256(args.p03_calibration_json)
                                    if getattr(args, "p03_mode", "none") == "adaptive_t2_25" else None),
        "tsnr": {
            "mode": getattr(args, "tsnr_mode", "none"),
            "beta": getattr(args, "tsnr_beta", 0.8),
            "grid_size": getattr(args, "tsnr_grid_size", 14),
            "causal_state": True,
            "state_input": "TI-processed per-image L1-normalized M1 image gradient",
            "probe_checkpoints": [60, 120, 178, 250, 298],
        } if getattr(args, "tsnr_mode", "none") != "none" else None,
        "p06_temporal_decomposition": {
            "enabled": bool(getattr(args, "p06_decomp_probe", False)),
            "beta": getattr(args, "p06_beta", 0.8),
            "perpendicular_scale": getattr(args, "p06_perp_scale", 0.5),
            "causal_reference": True,
            "reference": "bias-corrected EMA over past TI-processed per-image L1-normalized M1 gradients",
            "actual_attack_update": "unchanged FTM-M1",
            "estimators": ["M1", "EMA", "PARALLEL", "PERP", "HALF"],
            "probe_checkpoints": [60, 120, 178, 250, 298],
        } if getattr(args, "p06_decomp_probe", False) else None,
        "spw": {
            "enabled": bool(getattr(args, "use_spw", False)),
            "step_size": getattr(args, "spw_step_size", 0.05 / 299.0),
            "radius": getattr(args, "spw_radius", 0.05),
            "sample_count": 1,
            "parameter_selector": "all eligible non-normalization parameter groups; normalization modules excluded by type",
            "direction_normalization": "per_module_group_l2",
            "parameter_update": "theta <- theta - step_size * normalized_target_gradient",
            "step_calibration": "step_size = radius / 299 effective updates",
            "projection": "per_module_group_l2_ball_around_batch_entry_theta0; safety net rather than active saturation",
            "state_persistence": "across attack steps within one image batch",
            "first_effect_on_image_gradient": "next effective step",
            "batch_end_restore": "exact copy from immutable theta0 anchors",
            "clean_feature_cache": "recorded once at theta0 and kept fixed",
        } if getattr(args, "use_spw", False) else None,
        "p08_cross_space_reliability": {
            "mode": getattr(args, "csr_mode", "none"),
            "tau": getattr(args, "csr_tau", 1.0),
            "sample_count": 1,
            "history_state": False,
            "parameter_state": "read_only; source checkpoint never perturbed or updated",
            "parameter_gradient": "same backward as image and FTM tuning gradients",
            "feature_sensitivity": "L2 norm of active FTM tuning gradient divided by raw current same-layer feature L2 norm",
            "parameter_sensitivity": "same-layer Conv/Linear parameter-gradient L2 norm divided by parameter L2 norm",
            "feature_only_score": "log(feature_sensitivity + eps)",
            "cross_space_score": "log(feature_sensitivity + eps) - log(parameter_sensitivity + eps)",
            "weighting": "softmax(score/tau) over active layers of each surrogate, rescaled to mean one",
            "compute_control": "feature_only also requests same-layer parameter gradients but discards them",
        } if getattr(args, "csr_mode", "none") != "none" else None,
        "dlap_semantics": {
            "guidance_gradient_mode": ("computed_but_discarded" if args.dlap_control_mode == "repeatgrad_m2" else
                                       "computed_unused_directional_ema_radius_gradients_retained" if args.dlap_control_mode == "isotropic_sparse_m2" else
                                       f"{args.dlap_guidance_grad_norm}_ema" if args.dlap_alpha < 1.0 else "disabled"),
            "guidance_gradient_normalization_geometry": ("per_module_group_l2" if args.dlap_guidance_grad_norm == "normalized"
                                                         else "none"),
            "guidance_refresh_early": args.dlap_guidance_refresh,
            "guidance_refresh_late": args.dlap_late_guidance_refresh,
            "parameter_state_refresh_interval": args.dlap_state_refresh_interval,
            "parameter_gradient_samples_per_refresh": args.dlap_param_grad_samples_per_refresh,
            "direction_normalization": ("not_applied" if args.dlap_control_mode == "repeatgrad_m2" else
                                         "group_wise" if args.use_lap_adaptive_radius else "global"),
            "first_effective_step": ("no_parameter_perturbation" if args.dlap_control_mode == "repeatgrad_m2"
                                     else "zero_guidance_fixed_alpha" if args.dlap_control_mode in
                                     ("guidance_throughout_m1", "temporal_antithetic_guidance_m1")
                                     else "random_only"),
            "alpha_schedule": ("not_applied" if args.dlap_control_mode == "repeatgrad_m2" else
                               "inactive_isotropic" if args.dlap_alpha >= 1.0 else
                               "fixed_0.85_throughout" if args.dlap_control_mode in
                               ("guidance_throughout_m1", "temporal_antithetic_guidance_m1") else
                               "fixed_after_first_step" if args.dlap_disable_annealing else
                               "first_60_percent_then_linear"),
            "parameter_perturbation_phase": ("late_40_percent" if args.dlap_control_mode.startswith("late_")
                                             else "none" if args.dlap_control_mode == "repeatgrad_m2"
                                             else "all_steps"),
            "alpha_final": args.dlap_alpha,
            "rho_schedule": ("not_applied" if args.dlap_control_mode == "repeatgrad_m2" else
                             "exponential" if args.lap_rho_min < args.lap_rho else "constant"),
            "rho_initial": args.lap_rho,
            "rho_min": args.lap_rho_min,
            "samples_per_step_nominal": args.lap_num_samples,
            "sampling_schedule": ("M1_every_step" if args.dlap_control_mode in
                                  ("sampling_full_m1", "temporal_antithetic_m1",
                                   "guidance_throughout_m1", "temporal_antithetic_guidance_m1") else
                                  "M2_even_steps_M1_odd_steps" if args.dlap_control_mode == "sampling_interleave_k2" else
                                  "M2_every_fourth_step_M1_other_steps" if args.dlap_control_mode == "sampling_interleave_k4" else
                                  "nominal_or_existing_schedule"),
            "sample_interval": args.lap_num_samples_interval,
            "antithetic_exploration": args.dlap_control_mode in
                                     ("temporal_antithetic_m1", "temporal_antithetic_guidance_m1"),
            "rng_alignment_draw_discard": args.dlap_control_mode in
                                          ("temporal_antithetic_m1", "temporal_antithetic_guidance_m1"),
            "parameter_selector": args.lap_layer_selector,
            "adaptive_radius_requested": args.use_lap_adaptive_radius,
            "adaptive_radius_effective": (args.use_lap_adaptive_radius
                                          and args.dlap_control_mode != "repeatgrad_m2"),
            "parameter_perturbation_applied": args.dlap_control_mode != "repeatgrad_m2",
        } if args.use_dlap else None,
        "lap": {
            "lap_rho": args.lap_rho,
            "lap_rho_min": args.lap_rho_min,
            "lap_layer_selector": args.lap_layer_selector,
            "lap_num_samples": args.lap_num_samples,
            "lap_num_samples_interval": args.lap_num_samples_interval,
            "lap_grad_smooth_kernel": args.lap_grad_smooth_kernel,
            "use_lap_adaptive_radius": args.use_lap_adaptive_radius,
            "lap_beta": args.lap_beta,
            "lap_warmup": args.lap_warmup,
            "lap_cv_temp": args.lap_cv_temp,
            "lap_mode": args.lap_mode,
            "dlap_alpha": args.dlap_alpha,
            "dlap_disable_annealing": args.dlap_disable_annealing,
            "dlap_guidance_grad_norm": args.dlap_guidance_grad_norm,
            "dlap_guidance_refresh": args.dlap_guidance_refresh,
            "dlap_late_guidance_refresh": args.dlap_late_guidance_refresh,
            "dlap_state_refresh_interval": args.dlap_state_refresh_interval,
            "dlap_param_grad_samples_per_refresh": args.dlap_param_grad_samples_per_refresh,
        },
        "k_schedule": {
            "interval_active": interval_active,
        },
        "exp_settings": exp_settings,
        "args": vars(args),
    }


def save_attack_params(save_dir, params):
    path = os.path.join(save_dir, "attack_params.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=True, indent=2)
    return path


def load_attack_timing(save_dir):
    path = os.path.join(save_dir, "attack_timing.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_model_key(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def infer_whitebox_model(source_model_name, save_dir, target_model_names):
    if source_model_name:
        return source_model_name

    save_base_key = _normalize_model_key(os.path.basename(os.path.normpath(save_dir)))
    save_full_key = _normalize_model_key(os.path.normpath(save_dir))

    best_match = None
    best_len = -1
    for model_name in target_model_names:
        mk = _normalize_model_key(model_name)
        if not mk:
            continue
        if (mk in save_base_key) or (mk in save_full_key):
            if len(mk) > best_len:
                best_match = model_name
                best_len = len(mk)

    if best_match:
        return best_match

    if "ResNet50" in target_model_names:
        return "ResNet50"

    return None


def main(args):
    exp_settings = copy.deepcopy(exp_configuration[args.config_idx])

    # Allow command-line overrides so reviewer scripts can reuse one config entry.
    if args.max_iterations is not None:
        exp_settings["max_iterations"] = args.max_iterations
    if getattr(args, 'epsilon', None) is not None:
        exp_settings["epsilon"] = args.epsilon
    if args.num_images is not None:
        exp_settings["num_images"] = args.num_images
    if getattr(args, 'ftm_cache_clean_features_cpu', False):
        exp_settings["ftm_cache_clean_features_cpu"] = True
    if not args.model_weights_dir:
        args.model_weights_dir = "./vit_weight"

    run_generation = not args.eval
    run_evaluation = args.eval

    device = args.device
    policy = MemoryPolicy(args.vram_budget_gb, args.vram_warn_gb,
                          args.vram_strict, args.vram_monitor)
    generation_batch_size = args.batch_size
    evaluation_batch_size = args.eval_batch_size or args.batch_size
    if generation_batch_size < 1 or evaluation_batch_size < 1:
        raise ValueError("Generation and evaluation batch sizes must be positive")
    if run_evaluation and (args.vram_preflight or args.vram_preflight_report):
        raise ValueError("VRAM preflight flags apply to generation, not evaluation")
    batch_size = generation_batch_size if run_generation else evaluation_batch_size
    img_size = args.img_size

    if args.target_model_names.strip():
        exp_settings["target_model_names"] = [
            x.strip() for x in args.target_model_names.split(",") if x.strip()
        ]
    elif args.use_local_vit_targets:
        local_target_models = get_available_local_vit_models(args.model_weights_dir)
        if len(local_target_models) == 0:
            raise ValueError(
                f"No supported local ViT weights found in: {args.model_weights_dir}."
            )
        exp_settings["target_model_names"] = local_target_models

    if args.debug:
        exp_settings["num_images"] = 16
    if run_generation and args.dlap_control_mode == "repeatgrad_m2":
        if not args.use_dlap:
            raise ValueError("repeatgrad_m2 requires --use_dlap")
        if args.lap_num_samples != 2 or args.lap_num_samples_interval != 1:
            raise ValueError("repeatgrad_m2 requires M=2 and sample interval=1")
    if run_generation and args.use_spw:
        if (args.use_dlap or args.p02_diagnostic or args.p03_mode != "none"
                or args.tsnr_mode != "none" or args.p06_decomp_probe
                or args.csr_mode != "none"
                or args.attack_method != "RTMF" or args.model_name != "ResNet50"
                or not (args.batch_size == 32 or
                        (args.num_images == 5 and args.batch_size == 5))
                or args.momentum_decay != 1.0 or args.beta != 0.01
                or args.ensemble_size != 1 or args.config_idx != 1
                or args.max_iterations not in (None, 300)
                or args.epsilon not in (None, 16)
                or abs(args.spw_radius - 0.05) > 1e-12
                or (abs(args.spw_step_size) > 1e-12
                    and abs(args.spw_step_size - (0.05 / 299.0)) > 1e-12)):
            raise ValueError(
                "P0-7A-v2 SPW requires frozen ResNet50 RTMF, M=1, "
                "radius=0.05 and step_size=0 (null audit) or 0.05/299")

    if run_generation and args.csr_mode != "none":
        if (args.use_dlap or args.use_spw or args.p02_diagnostic
                or args.p03_mode != "none" or args.tsnr_mode != "none"
                or args.p06_decomp_probe
                or args.attack_method != "RTMF" or args.model_name != "ResNet50"
                or not (args.batch_size == 32 or
                        (args.num_images == 5 and args.batch_size == 5))
                or args.momentum_decay != 1.0 or args.beta != 0.01
                or args.ensemble_size != 1 or args.config_idx != 1
                or args.max_iterations not in (None, 300)
                or args.epsilon not in (None, 16)
                or abs(args.csr_tau - 1.0) > 1e-12):
            raise ValueError(
                "P0-8 CSR requires frozen ResNet50 RTMF, M=1, tau=1.0, "
                "batch32 (or five-image smoke batch5), with no other mechanism enabled")

    if run_generation and args.tsnr_mode != "none":
        if (args.use_dlap or args.p02_diagnostic or args.p03_mode != "none"
                or args.attack_method != "RTMF" or args.model_name != "ResNet50"
                or not (args.batch_size == 32 or
                        (args.num_images == 5 and args.batch_size == 5))
                or args.momentum_decay != 1.0 or args.beta != 0.01
                or args.ensemble_size != 1 or args.config_idx != 1
                or args.max_iterations not in (None, 300)
                or args.epsilon not in (None, 16)
                or abs(args.tsnr_beta - 0.8) > 1e-12
                or args.tsnr_grid_size != 14):
            raise ValueError("First-stage TSNR requires frozen ResNet50 RTMF, beta=0.8, grid=14")

    if run_generation and args.p06_decomp_probe:
        if (args.use_dlap or args.p02_diagnostic or args.p03_mode != "none"
                or args.tsnr_mode != "none"
                or args.attack_method != "RTMF" or args.model_name != "ResNet50"
                or not (args.batch_size == 32 or
                        (args.num_images == 5 and args.batch_size == 5))
                or args.momentum_decay != 1.0 or args.beta != 0.01
                or args.ensemble_size != 1 or args.config_idx != 1
                or args.max_iterations not in (None, 300)
                or args.epsilon not in (None, 16)
                or abs(args.p06_beta - 0.8) > 1e-12
                or abs(args.p06_perp_scale - 0.5) > 1e-12):
            raise ValueError(
                "P0-6 requires frozen ResNet50 RTMF, beta=0.8, perpendicular scale=0.5")

    if run_generation and args.p03_mode != "none":
        if (args.use_dlap or args.p02_diagnostic or args.attack_method != "RTMF"
                or args.model_name != "ResNet50"
                or not (args.batch_size == 32 or
                        (args.num_images == 5 and args.batch_size == 5))
                or args.momentum_decay != 1.0 or args.beta != 0.01
                or args.ensemble_size != 1 or args.config_idx != 1
                or args.max_iterations not in (None, 300)
                or args.epsilon not in (None, 16)):
            raise ValueError("P0-3 clean RepeatGrad requires the frozen ResNet50 RTMF setting")
        if args.p03_mode == "adaptive_t2_25" and not args.p03_calibration_json:
            raise ValueError("Adaptive P0-3 requires frozen calibration JSON")
        if args.p03_mode != "adaptive_t2_25" and args.p03_calibration_json:
            raise ValueError("Only adaptive P0-3 may take a calibration JSON")
    efficient_modes = ("late_m2_warmk4", "late_m1_warmk4", "late_m1_sparsek4")
    if run_generation and args.dlap_control_mode in efficient_modes:
        expected_m = 2 if args.dlap_control_mode == "late_m2_warmk4" else 1
        expected_late_k = 4 if args.dlap_control_mode == "late_m1_sparsek4" else 1
        if (not args.use_dlap or args.lap_num_samples != expected_m
                or args.lap_num_samples_interval != 1
                or args.dlap_guidance_grad_norm != "normalized"
                or args.dlap_guidance_refresh != 4
                or args.dlap_late_guidance_refresh != expected_late_k
                or args.dlap_alpha >= 1.0 or args.dlap_disable_annealing):
            raise ValueError("Efficient D-LAP mode has inconsistent fixed settings")
    if run_generation and args.dlap_control_mode in ("sparse_state_m2", "isotropic_sparse_m2"):
        isotropic_control = args.dlap_control_mode == "isotropic_sparse_m2"
        if isotropic_control and (args.dlap_state_refresh_interval != 4
                                  or args.dlap_param_grad_samples_per_refresh != 1):
            raise ValueError("Isotropic-M2 requires frozen sparse K4/S1")
        if (not args.use_dlap or args.lap_num_samples != 2
                or args.lap_num_samples_interval != 1
                or args.dlap_guidance_grad_norm != "normalized"
                or not args.use_lap_adaptive_radius
                or (args.dlap_alpha != 1.0 if isotropic_control else args.dlap_alpha >= 1.0)
                or args.dlap_disable_annealing
                or args.dlap_guidance_refresh != 1 or args.dlap_late_guidance_refresh != 1
                or args.dlap_state_refresh_interval not in (1, 2, 4)
                or args.dlap_param_grad_samples_per_refresh not in (1, 2)):
            raise ValueError("Sparse-state D-LAP mode has inconsistent fixed settings")
    sampling_modes = ("sampling_full_m1", "sampling_interleave_k2", "sampling_interleave_k4")
    temporal_modes = ("temporal_antithetic_m1", "guidance_throughout_m1",
                      "temporal_antithetic_guidance_m1")
    if run_generation and args.dlap_control_mode in sampling_modes + temporal_modes:
        expected_m = 1 if args.dlap_control_mode == "sampling_full_m1" or args.dlap_control_mode in temporal_modes else 2
        if (not args.use_dlap or args.lap_num_samples != expected_m
                or args.lap_num_samples_interval != 1
                or args.dlap_guidance_grad_norm != "normalized"
                or not args.use_lap_adaptive_radius or args.dlap_alpha >= 1.0
                or args.dlap_disable_annealing
                or args.dlap_guidance_refresh != 1 or args.dlap_late_guidance_refresh != 1
                or args.dlap_state_refresh_interval != 4
                or args.dlap_param_grad_samples_per_refresh != 1):
            raise ValueError("P0-1D sampling mode has inconsistent B5 settings")

    print(today_string, flush=True)
    print(args, flush=True)
    print(exp_settings, flush=True)
    print("#" * 50, flush=True)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    target_model_names = exp_settings["target_model_names"]
    small_sized_models = [
        "vit_base_patch16_224",
        "levit_384",
        "convit_base",
        "twins_svt_base",
        "pit",
    ] + LOCAL_VIT_MODEL_NAMES
    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    trn = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    generation_metadata = None
    manifest_records = None
    manifest_path = os.path.join(args.save_dir, "sample_manifest.csv")
    params_path = os.path.join(args.save_dir, "attack_params.json")
    if run_evaluation and os.path.isfile(params_path):
        with open(params_path, "r", encoding="utf-8") as stream:
            generation_metadata = json.load(stream)

    if run_evaluation and os.path.isfile(manifest_path):
        if generation_metadata is None:
            raise ValueError("Generation manifest exists without attack_params.json")
        manifest_records = load_sample_manifest(manifest_path, generation_metadata)
        image_id_list = [r["filename"] for r in manifest_records]
        label_ori_list = [int(r["true_label"]) for r in manifest_records]
        label_tar_list = [int(r["target_label"]) for r in manifest_records]
        total_img_num = len(manifest_records)
        source_model_name = generation_metadata["source_model_name"]
        saved_size = generation_metadata.get("args", {}).get("img_size", img_size)
        if img_size != saved_size:
            print(f"[Eval] Using generation img_size={saved_size} from attack_params.json", flush=True)
            img_size = saved_size
            trn = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        if not args.target_model_names.strip() and not args.use_local_vit_targets:
            target_model_names = generation_metadata["exp_settings"]["target_model_names"]
        print(f"[Eval] Using generation manifest: {manifest_path}", flush=True)
    else:
        if run_evaluation:
            if generation_metadata is not None and generation_metadata.get("protocol_version") == "p0a-1":
                raise FileNotFoundError(f"New run is missing its generation manifest: {manifest_path}")
            print("WARNING: legacy evaluation without generation manifest; source may be inferred from path", flush=True)
        if run_generation:
            if args.sample_manifest_input:
                frozen_rows = load_generation_manifest_input(
                    args.sample_manifest_input, seed, args.model_name,
                    exp_settings["num_images"]
                )
                sample_rows = [
                    (row["filename"], int(row["true_label"]), int(row["target_label"]),
                     int(row["raw_true_label"]), int(row["raw_target_label"]))
                    for row in frozen_rows
                ]
                label_metadata = {
                    "true_label_shifted": any(row[1] != row[3] for row in sample_rows),
                    "target_label_shifted": any(row[2] != row[4] for row in sample_rows),
                }
                print(f"[Generate] Reusing frozen sample manifest: {args.sample_manifest_input}", flush=True)
            else:
                image_id_list, label_ori_list, label_tar_list, label_metadata = load_ground_truth(
                    args.image_csv, return_metadata=True
                )
                sample_rows = list(zip(image_id_list, label_ori_list, label_tar_list,
                                       label_metadata["raw_true_labels"],
                                       label_metadata["raw_target_labels"]))
                if seed != 42:
                    random.shuffle(sample_rows)
                sample_rows = sample_rows[:min(exp_settings["num_images"], len(sample_rows))]
            image_id_list = [r[0] for r in sample_rows]
            label_ori_list = [r[1] for r in sample_rows]
            label_tar_list = [r[2] for r in sample_rows]
            raw_ori_list = [r[3] for r in sample_rows]
            raw_tar_list = [r[4] for r in sample_rows]
            total_img_num = len(sample_rows)
            if args.vram_preflight and total_img_num != generation_batch_size:
                raise ValueError("Preflight must execute exactly one full generation batch")
            if (args.vram_strict and not args.vram_preflight
                    and total_img_num > generation_batch_size
                    and not args.vram_preflight_report):
                raise ValueError("Strict full generation requires --vram_preflight_report from one matching batch")
            validate_samples([
                {"filename": image_id_list[i], "true_label": label_ori_list[i],
                 "target_label": label_tar_list[i]} for i in range(total_img_num)
            ])
            print(f"[Generate] Label shifts: true={label_metadata['true_label_shifted']}, "
                  f"target={label_metadata['target_label_shifted']}", flush=True)
        else:
            if run_evaluation and generation_metadata is not None and args.num_images is None:
                saved_count = generation_metadata.get("num_images")
                if isinstance(saved_count, int):
                    exp_settings["num_images"] = saved_count
                    print(f"[Eval] Historical image count from attack_params.json: {saved_count}", flush=True)
            image_id_list, label_ori_list, label_tar_list = load_ground_truth(args.image_csv)
            total_img_num = min(exp_settings["num_images"], len(image_id_list))
            image_id_list = image_id_list[:total_img_num]
            label_ori_list = label_ori_list[:total_img_num]
            label_tar_list = label_tar_list[:total_img_num]

    source_models = []
    memory_tracker = CudaMemoryTracker(device, policy)
    if run_generation:
        memory_tracker.start()
    generation_sig = None
    validated_preflight = None
    if manifest_records is None:
        source_model_name = (
            generation_metadata.get("args", {}).get("model_name")
            if run_evaluation and generation_metadata is not None else None
        )
        if run_evaluation and source_model_name:
            print(f"[Eval] Historical source model from attack_params.json: {source_model_name}", flush=True)

    if run_generation:
        atk_type = args.attack_method
        exp_settings["ftm_beta"] = args.beta
        exp_settings["ftm_ensemble_size"] = args.ensemble_size

        check_new_run_directory(args.save_dir)

        source_model_name = args.model_name
        print("Loading source model", source_model_name, flush=True)
        source_model = WrapperModel(
            load_model(
                source_model_name,
                model_weights_dir=args.model_weights_dir,
                prefer_local_weights=args.prefer_local_weights,
            ),
            mean,
            stddev,
            source_model_name in small_sized_models,
        ).to(device)
        source_model = source_model.eval()
        source_models = [source_model]
        source_weight_info = source_model.model._codex_weight_info
        print(f"[Generate] Source weights: {json.dumps(source_weight_info, ensure_ascii=True)}", flush=True)
        generation_sig = generation_signature(args, exp_settings, source_weight_info,
                                              memory_tracker.gpu_name)
        memory_tracker.snapshot("after_source_load")
        if args.vram_preflight_report:
            validated_preflight = validate_preflight(
                args.vram_preflight_report, generation_sig, policy,
                generation_batch_size, memory_tracker.free_at_start_bytes)
            print("[VRAM] Matching preflight accepted: peak allocated "
                  f"{validated_preflight['peak_allocated']['gib']:.3f} GiB, "
                  f"reserved {validated_preflight['peak_reserved']['gib']:.3f} GiB", flush=True)

        os.makedirs(args.save_dir, exist_ok=True)

        if exp_settings["ftm_ensemble_size"] > 1:
            for _ in range(exp_settings["ftm_ensemble_size"] - 1):
                source_models.append(copy.deepcopy(source_model))

        manifest_records = [
            {"generation_index": i, "filename": image_id_list[i],
             "true_label": label_ori_list[i], "target_label": label_tar_list[i],
             "raw_true_label": raw_ori_list[i], "raw_target_label": raw_tar_list[i],
             "seed": seed, "source_model": source_model_name}
            for i in range(total_img_num)
        ]
        save_sample_manifest(args.save_dir, manifest_records)
        params_path = save_attack_params(
            args.save_dir,
            build_attack_params(args, exp_settings, total_img_num, atk_type,
                                source_model_name, source_weight_info,
                                manifest_path,
                                {"true_label_shifted": label_metadata["true_label_shifted"],
                                 "target_label_shifted": label_metadata["target_label_shifted"]}),
        )
        print(f"[Generate] Attack params and manifest saved to {args.save_dir}", flush=True)
    else:
        print(
            f"[Eval] Reusing adversarial images from {os.path.join(args.save_dir, 'adv_imgs')}",
            flush=True,
        )

    num_batches = int(np.ceil(len(image_id_list) / batch_size))
    num_images = 0
    progress_log_interval = 50
    generation_last_logged = 0
    adv_dir = os.path.join(args.save_dir, "adv_imgs")
    if run_generation:
        os.makedirs(adv_dir, exist_ok=True)
    elif not os.path.isdir(adv_dir):
        raise FileNotFoundError(f"Adversarial image directory is missing: {adv_dir}")
    adv_file_paths = [None] * total_img_num
    attack_total_elapsed = 0.0
    attack_num_images = 0
    attack_metrics = None
    peak_cuda_memory_bytes = 0
    compute_totals = {key: 0 for key in COMPUTE_KEYS}
    compute_batches = []

    generation_total_elapsed = 0.0
    if run_generation:
        generation_start_time = time.time()
        use_cuda_timing = str(device).startswith("cuda") and torch.cuda.is_available()
        for batch_idx in range(num_batches):
            batch_size_cur = min(batch_size, len(image_id_list) - batch_idx * batch_size)
            img_start = batch_idx * batch_size + 1
            img_end = batch_idx * batch_size + batch_size_cur
            print(
                f"[Batch {batch_idx + 1}/{num_batches}] images {img_start}-{img_end} ...",
                flush=True,
            )

            img = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
            for offset in range(batch_size_cur):
                image_name = image_id_list[batch_idx * batch_size + offset]
                image_path = resolve_image_path(args.image_dir, image_name)
                img[offset] = trn(Image.open(image_path).convert("RGB"))

            labels = torch.tensor(
                label_ori_list[batch_idx * batch_size : batch_idx * batch_size + batch_size_cur]
            ).to(device)
            target_labels = torch.tensor(
                label_tar_list[batch_idx * batch_size : batch_idx * batch_size + batch_size_cur]
            ).to(device)

            if use_cuda_timing:
                torch.cuda.synchronize()
            memory_tracker.snapshot("before_attack", batch_idx)
            attack_batch_start = time.perf_counter()
            batch_compute = {key: 0 for key in COMPUTE_KEYS}
            batch_compute["batch_index"] = batch_idx
            batch_compute["batch_size"] = batch_size_cur
            def memory_checkpoint(name, step=None):
                memory_tracker.attack_checkpoint(name, batch_idx, step)

            if args.csr_mode != "none":
                adv_imgs = cross_space_reliability_attack(
                    source_models, img, target_labels, exp_settings, device,
                    mode=args.csr_mode, tau=args.csr_tau, attack_type=atk_type,
                    num_iter=exp_settings.get("max_iterations", 300),
                    max_epsilon=exp_settings.get("epsilon", 16),
                    mu=args.momentum_decay,
                    compute_stats=batch_compute,
                    memory_checkpoint=memory_checkpoint if args.vram_monitor else None,
                )
            elif args.use_spw:
                adv_imgs = stateful_param_walk_attack(
                    source_models, img, target_labels, exp_settings, device,
                    attack_type=atk_type,
                    num_iter=exp_settings.get("max_iterations", 300),
                    max_epsilon=exp_settings.get("epsilon", 16),
                    mu=args.momentum_decay,
                    step_size=args.spw_step_size, radius=args.spw_radius,
                    compute_stats=batch_compute,
                    memory_checkpoint=memory_checkpoint if args.vram_monitor else None,
                )
            elif args.p06_decomp_probe:
                adv_imgs = temporal_decomp_probe_attack(
                    source_models, img, target_labels, exp_settings, device,
                    attack_type=atk_type,
                    num_iter=exp_settings.get("max_iterations", 300),
                    max_epsilon=exp_settings.get("epsilon", 16),
                    mu=args.momentum_decay, beta=args.p06_beta,
                    perp_scale=args.p06_perp_scale, compute_stats=batch_compute,
                    memory_checkpoint=memory_checkpoint if args.vram_monitor else None,
                    save_dir=args.save_dir, batch_index=batch_idx,
                    save_probe_artifacts=not args.vram_preflight,
                )
            elif args.tsnr_mode != "none":
                adv_imgs = tsnr_ftm_attack(
                    source_models, img, target_labels, exp_settings, device,
                    mode=args.tsnr_mode, attack_type=atk_type,
                    num_iter=exp_settings.get("max_iterations", 300),
                    max_epsilon=exp_settings.get("epsilon", 16),
                    mu=args.momentum_decay, beta=args.tsnr_beta,
                    grid_size=args.tsnr_grid_size, compute_stats=batch_compute,
                    memory_checkpoint=memory_checkpoint if args.vram_monitor else None,
                    save_dir=args.save_dir, batch_index=batch_idx,
                )
            elif args.p03_mode != "none":
                threshold = None
                if args.p03_mode == "adaptive_t2_25":
                    with open(args.p03_calibration_json, encoding="utf-8") as stream:
                        calibration = json.load(stream)
                    if (calibration.get("quantile") != 0.25 or
                            calibration.get("calibration_sample_count") != 598 or
                            calibration.get("rule") != "t2_batch_mean_l1 <= threshold"):
                        raise ValueError("P0-3 adaptive calibration metadata differs from frozen rule")
                    threshold = float(calibration["threshold"])
                adv_imgs = clean_repeatgrad_attack(
                    source_models, img, target_labels, exp_settings, device,
                    args.p03_mode, threshold=threshold,
                    num_iter=exp_settings.get("max_iterations", 300),
                    max_epsilon=exp_settings.get("epsilon", 16),
                    mu=args.momentum_decay, compute_stats=batch_compute,
                    memory_checkpoint=memory_checkpoint if args.vram_monitor else None,
                    save_dir=args.save_dir, batch_index=batch_idx,
                    validate_parameters=args.repeatgrad_validate,
                )
            elif args.use_dlap:
                diagnostic_observer = None
                if args.p02_diagnostic:
                    from p02_diagnostics import P02Observer
                    diagnostic_observer = P02Observer(
                        args.save_dir, batch_idx, args.dlap_control_mode,
                        target_labels, device, args.momentum_decay)
                adv_imgs = dlap_ftm_attack(
                    source_models,
                    img,
                    labels,
                    target_labels,
                    exp_settings,
                    device,
                    atk_type,
                    num_iter=exp_settings.get("max_iterations", 300),
                    max_epsilon=exp_settings.get("epsilon", 16),
                    mu=args.momentum_decay,
                    lap_rho=args.lap_rho,
                    lap_rho_min=args.lap_rho_min,
                    lap_layer_selector=args.lap_layer_selector,
                    lap_num_samples=args.lap_num_samples,
                    lap_num_samples_interval=args.lap_num_samples_interval,
                    lap_grad_smooth_kernel=args.lap_grad_smooth_kernel,
                    use_lap_adaptive_radius=args.use_lap_adaptive_radius,
                    lap_beta=args.lap_beta,
                    lap_warmup=args.lap_warmup,
                    lap_cv_temp=args.lap_cv_temp,
                    lap_mode=args.lap_mode,
                    dlap_alpha=args.dlap_alpha,
                    dlap_disable_annealing=args.dlap_disable_annealing,
                    control_mode=args.dlap_control_mode,
                    repeatgrad_validate=args.repeatgrad_validate,
                    dlap_guidance_grad_norm=args.dlap_guidance_grad_norm,
                    dlap_guidance_refresh=args.dlap_guidance_refresh,
                    dlap_late_guidance_refresh=args.dlap_late_guidance_refresh,
                    dlap_state_refresh_interval=args.dlap_state_refresh_interval,
                    dlap_param_grad_samples_per_refresh=args.dlap_param_grad_samples_per_refresh,
                    sparse_state_validate=args.sparse_state_validate,
                    dlap_restore_validate=args.dlap_restore_validate,
                    compute_stats=batch_compute,
                    memory_checkpoint=memory_checkpoint if args.vram_monitor else None,
                    diagnostic_observer=diagnostic_observer,
                )
                if diagnostic_observer is not None:
                    diagnostic_observer.finish()
            else:
                adv_imgs = ftm_attack(
                    source_models,
                    img,
                    labels,
                    target_labels,
                    exp_settings,
                    device,
                    atk_type,
                    num_iter=exp_settings.get("max_iterations", 300),
                    max_epsilon=exp_settings.get("epsilon", 16),
                    mu=args.momentum_decay,
                    compute_stats=batch_compute,
                    memory_checkpoint=memory_checkpoint if args.vram_monitor else None,
                )

            for key in COMPUTE_KEYS:
                compute_totals[key] += batch_compute[key]
            compute_batches.append(batch_compute)

            if use_cuda_timing:
                torch.cuda.synchronize()
            memory_tracker.snapshot("after_attack", batch_idx)
            peak_cuda_memory_bytes = max(peak_cuda_memory_bytes, memory_tracker.peak()[0])
            attack_total_elapsed += time.perf_counter() - attack_batch_start
            attack_num_images += batch_size_cur
            num_images += batch_size_cur

            for offset in range(batch_size_cur):
                img_idx = batch_idx * batch_size + offset
                save_name = build_adv_filename(
                    img_idx,
                    image_id_list[img_idx],
                    label_ori_list[img_idx],
                    label_tar_list[img_idx],
                )
                save_path = os.path.join(adv_dir, save_name)
                save_images(adv_imgs[offset], save_path)
                adv_file_paths[img_idx] = save_path

            if should_log_by_interval(
                num_images, generation_last_logged, progress_log_interval, total_img_num
            ):
                generation_elapsed = time.time() - generation_start_time
                generation_eta = (
                    (generation_elapsed / num_images) * (total_img_num - num_images)
                    if num_images > 0
                    else 0.0
                )
                print(
                    f"\n[Generate] Saved [{num_images}/{total_img_num}] adversarial images to {adv_dir} "
                    f"| elapsed {format_hms(generation_elapsed)} | eta {format_hms(generation_eta)}\n",
                    flush=True,
                )
                generation_last_logged = num_images

            del img, labels, target_labels, adv_imgs, batch_compute

        generation_total_elapsed = time.time() - generation_start_time
        memory_tracker.snapshot("after_generation")
        attack_metrics = build_attack_timing_metrics(
            attack_total_elapsed, attack_num_images, peak_cuda_memory_bytes
        )
        timing_path = save_attack_timing(args.save_dir, attack_metrics)
        compute_path = save_compute_stats(args.save_dir, compute_totals, compute_batches)
        del source_models, source_model
        gc.collect()
        if use_cuda_timing:
            torch.cuda.empty_cache()
        memory_tracker.snapshot("after_model_cleanup")
        memory_report = memory_tracker.report(
            "preflight" if args.vram_preflight else "generation", generation_sig,
            generation_batch_size, evaluation_batch_size, attack_num_images,
            args.vram_preflight_report or None)
        memory_report["source_model_name"] = source_model_name
        memory_report["attack_type"] = atk_type
        memory_report["likely_consumers"] = likely_memory_consumers(
            atk_type, use_dlap=args.use_dlap,
            ensemble_size=exp_settings["ftm_ensemble_size"],
            use_spw=args.use_spw, use_csr=(args.csr_mode != "none"))
        memory_path = os.path.join(args.save_dir, "memory_stats.json")
        with open(memory_path, "w", encoding="utf-8") as stream:
            json.dump(memory_report, stream, indent=2)
        print("[VRAM] GPU={} total={:.3f} GiB peak_allocated={:.3f} GiB "
              "peak_reserved={:.3f} GiB before_attack={:.3f} GiB "
              "after_attack={:.3f} GiB state={} batch={}".format(
                  memory_report["gpu_model"], memory_report["total_vram"]["gib"],
                  memory_report["peak_allocated"]["gib"],
                  memory_report["peak_reserved"]["gib"],
                  memory_report["allocated_before_attack"]["gib"],
                  memory_report["allocated_after_attack"]["gib"],
                  memory_report["warning_state"], generation_batch_size), flush=True)
        if memory_report["warning_state"] == "warning":
            print("[VRAM] Warning: peak allocated reached the preferred "
                  f"{policy.warn_gb:.3f} GiB threshold.", flush=True)
        elif memory_report["warning_state"] == "unsafe":
            print("[VRAM] Unsafe: peak allocated reached the configured budget; "
                  "use an explicit smaller batch for a new preflight. Likely consumers: "
                  + ", ".join(memory_report["likely_consumers"]), flush=True)
        print(f"[Generate] Attack timing saved to {timing_path}", flush=True)
        print(f"[Generate] Compute counters saved to {compute_path}", flush=True)
        print(f"[Generate] Memory telemetry saved to {memory_path}", flush=True)
        print(f"[Generate] Compute totals: {json.dumps(compute_totals)}", flush=True)
        print(
            f"\n[Generate] Complete | total elapsed {format_hms(generation_total_elapsed)}\n",
            flush=True,
        )
        if args.vram_preflight and policy.strict and memory_report["warning_state"] == "unsafe":
            raise RuntimeError("Strict VRAM preflight exceeded budget; do not start a full run")
    else:
        print("[Generate] Skipped by --eval; loading existing adversarial images.", flush=True)
        attack_metrics = load_attack_timing(args.save_dir)
        compute_path = os.path.join(args.save_dir, "compute_stats.json")
        if os.path.isfile(compute_path):
            with open(compute_path, "r", encoding="utf-8") as stream:
                compute_totals = json.load(stream)["totals"]
        elif generation_metadata is not None and generation_metadata.get("protocol_version") == "p0a-1":
            raise FileNotFoundError(f"New run is missing compute counters: {compute_path}")
        else:
            compute_totals = None
        for img_idx in range(total_img_num):
            save_name = build_adv_filename(
                img_idx,
                image_id_list[img_idx],
                label_ori_list[img_idx],
                label_tar_list[img_idx],
            )
            adv_file_paths[img_idx] = os.path.join(adv_dir, save_name)
            if manifest_records is not None and not os.path.isfile(adv_file_paths[img_idx]):
                raise FileNotFoundError(f"Manifest image is missing: {adv_file_paths[img_idx]}")
        if manifest_records is not None:
            expected_names = {os.path.basename(path) for path in adv_file_paths}
            actual_names = {name for name in os.listdir(adv_dir) if name.lower().endswith(".png")}
            if actual_names != expected_names:
                raise ValueError("Adversarial image set differs from the generation manifest")

    evaluation_total_elapsed = 0.0
    eval_results = []
    if run_evaluation:
        eval_results_path = os.path.join(args.save_dir, "eval_results.txt")
        eval_model_info_path = os.path.join(args.save_dir, "evaluation_model_weights.json")
        eval_config_path = os.path.join(args.save_dir, "evaluation_config.json")
        eval_memory_path = os.path.join(args.save_dir, "evaluation_memory_stats.json")
        if any(os.path.exists(path) for path in (eval_results_path, eval_model_info_path,
                                                 eval_config_path, eval_memory_path)):
            raise FileExistsError(f"Refusing to overwrite existing evaluation artifacts in {args.save_dir}")
        eval_results = [EvalResult(x) for x in target_model_names]
        eval_model_info = []

        memory_tracker.start()
        evaluation_start_time = time.time()
        for eval_result, target_model_name in zip(eval_results, target_model_names):
            print(f"Loading target model {target_model_name}...", flush=True)
            target_model = WrapperModel(
                load_model(
                    target_model_name,
                    model_weights_dir=args.model_weights_dir,
                    prefer_local_weights=args.prefer_local_weights,
                ),
                mean,
                stddev,
                target_model_name in small_sized_models,
            ).to(device)
            target_model.eval()
            weight_info = target_model.model._codex_weight_info
            eval_model_info.append({"model_name": target_model_name, "weights": weight_info})
            print(f"[Eval] Target weights: {json.dumps(weight_info, ensure_ascii=True)}", flush=True)
            memory_tracker.snapshot("before_target", len(eval_model_info) - 1)

            for batch_idx in range(num_batches):
                batch_size_cur = min(batch_size, total_img_num - batch_idx * batch_size)
                adv_batch = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)

                for offset in range(batch_size_cur):
                    img_idx = batch_idx * batch_size + offset
                    adv_batch[offset] = trn(Image.open(adv_file_paths[img_idx]).convert("RGB"))

                target_labels = torch.tensor(
                    label_tar_list[batch_idx * batch_size : batch_idx * batch_size + batch_size_cur]
                ).to(device)
                with torch.no_grad():
                    pred_labels = target_model(adv_batch)
                    pred_labels = torch.argmax(pred_labels, dim=1)
                    eval_result.update(pred_labels, target_labels)
                del adv_batch, target_labels, pred_labels

            del target_model
            gc.collect()
            if memory_tracker.enabled:
                torch.cuda.empty_cache()
            memory_tracker.snapshot("after_target_cleanup", len(eval_model_info) - 1)

        evaluation_total_elapsed = time.time() - evaluation_start_time
        memory_tracker.snapshot("after_model_cleanup")
        evaluation_memory = memory_tracker.report(
            "evaluation", None,
            generation_metadata.get("generation_batch_size", generation_metadata.get("args", {}).get("batch_size"))
            if generation_metadata else None,
            evaluation_batch_size, total_img_num)
        with open(eval_memory_path, "w", encoding="utf-8") as stream:
            json.dump(evaluation_memory, stream, indent=2)
        with open(eval_config_path, "w", encoding="utf-8") as stream:
            json.dump({"timestamp": datetime.now().isoformat(), "args": vars(args),
                       "target_model_names": target_model_names,
                       "source_model_name": source_model_name,
                       "generation_batch_size": evaluation_memory["generation_batch_size"],
                       "evaluation_batch_size": evaluation_batch_size,
                       "memory_policy": policy.metadata(),
                       "peak_allocated": evaluation_memory["peak_allocated"],
                       "peak_reserved": evaluation_memory["peak_reserved"],
                       "memory_warning_state": evaluation_memory["warning_state"],
                       "sample_manifest": os.path.abspath(manifest_path) if manifest_records is not None else None,
                       "environment": environment_metadata(device)}, stream, indent=2)
        with open(eval_model_info_path, "w", encoding="utf-8") as stream:
            json.dump(eval_model_info, stream, indent=2)
        print(
            f"\n[Evaluate] Complete | total elapsed {format_hms(evaluation_total_elapsed)}\n",
            flush=True,
        )
        print("[VRAM] Evaluation peak allocated {:.3f} GiB, reserved {:.3f} GiB; "
              "batch={} state={}".format(evaluation_memory["peak_allocated"]["gib"],
                                          evaluation_memory["peak_reserved"]["gib"],
                                          evaluation_batch_size,
                                          evaluation_memory["warning_state"]), flush=True)
        for eval_result in eval_results:
            print(eval_result, flush=True)

    if run_evaluation:
        finish_string = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
        whitebox_model = infer_whitebox_model(
            source_model_name, args.save_dir, target_model_names
        )
        has_whitebox_in_targets = (
            any(r.model_name == whitebox_model for r in eval_results)
            if whitebox_model
            else False
        )
        transfer_eval_results = (
            [r for r in eval_results if r.model_name != whitebox_model]
            if has_whitebox_in_targets
            else list(eval_results)
        )
        vit_set = set(small_sized_models)

        def get_rate(result):
            return float(getattr(result, "success_rate", 0.0))

        with open(eval_results_path, "w", encoding="utf-8") as f:
            f.write(f"Evaluation Results ({finish_string})\n")
            f.write(f"Job started at {today_string}\n")
            f.write(f"Generation elapsed: {format_hms(generation_total_elapsed)}\n")
            if attack_metrics is not None:
                f.write(
                    f"Attack elapsed: {attack_metrics['attack_elapsed_hms']} "
                    f"({attack_metrics['attack_elapsed_seconds']:.2f}s)\n"
                )
            if compute_totals is not None:
                f.write(f"Compute counters: {json.dumps(compute_totals, sort_keys=True)}\n")
            f.write("-" * 50 + "\n")

            for eval_result in eval_results:
                f.write(str(eval_result) + "\n")

            f.write("-" * 50 + "\n")
            if has_whitebox_in_targets:
                f.write(f"Excluded White-box Model: {whitebox_model}\n")
            else:
                f.write(
                    "Excluded White-box Model: NONE (not found; averages include all targets)\n"
                )

            cnn_rates = [get_rate(r) for r in transfer_eval_results if r.model_name not in vit_set]
            vit_rates = [get_rate(r) for r in transfer_eval_results if r.model_name in vit_set]
            all_rates = [get_rate(r) for r in transfer_eval_results]

            if cnn_rates:
                f.write(
                    f"CNN  Average Attack Success: {sum(cnn_rates)/len(cnn_rates):.1f}% "
                    f"({len(cnn_rates)} models)\n"
                )
            if vit_rates:
                f.write(
                    f"ViT  Average Attack Success: {sum(vit_rates)/len(vit_rates):.1f}% "
                    f"({len(vit_rates)} models)\n"
                )
            if all_rates:
                f.write(
                    f"All  Average Attack Success: {sum(all_rates)/len(all_rates):.1f}% "
                    f"({len(all_rates)} models)\n"
                )


def argument_parsing():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--eval", action="store_true")
    is_eval = pre_parser.parse_known_args()[0].eval

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Evaluation-only batch size; defaults to --batch_size.")
    parser.add_argument("--vram_budget_gb", type=float, default=10.0)
    parser.add_argument("--vram_warn_gb", type=float, default=9.5)
    parser.add_argument("--vram_monitor", action="store_true",
                        help="Capture bounded attack-step memory checkpoints.")
    parser.add_argument("--vram_strict", action="store_true",
                        help="Require a matching safe preflight before a multi-batch generation run.")
    parser.add_argument("--vram_preflight", action="store_true",
                        help="Run one full batch and save a reusable memory report.")
    parser.add_argument("--vram_preflight_report", type=str, default="",
                        help="Completed matching memory_stats.json for a strict full run.")
    parser.add_argument("--save_dir", type=str, default="./exp/test")
    parser.add_argument("--image_csv", type=str, default="./data/ILSVRC2012/labels.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=299)
    parser.add_argument("--config_idx", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_weights_dir", type=str, default="./vit_weight")
    parser.add_argument("--prefer_local_weights", action="store_true")
    parser.add_argument("--use_local_vit_targets", action="store_true")
    parser.add_argument("--target_model_names", type=str, default="")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override config max_iterations.")
    parser.add_argument("--num_images", type=int, default=None)

    if not is_eval:
        parser.add_argument("--model_name", type=str, default="ResNet50")
        parser.add_argument("--attack_method", type=str, default="RTMF")
        parser.add_argument("--beta", type=float, default=0.01)
        parser.add_argument("--ftm_cache_clean_features_cpu", action="store_true")
        parser.add_argument("--ensemble_size", type=int, default=1)
        parser.add_argument("--momentum_decay", type=float, default=1.0)
        parser.add_argument("--image_dir", type=str, default="./data/ILSVRC2012/images")
        parser.add_argument("--epsilon", type=int, default=None)
        parser.add_argument("--sample_manifest_input", type=str, default="")
        parser.add_argument("--experiment_condition", type=str, default="")

        parser.add_argument("--use_dlap", action="store_true")
        parser.add_argument("--dlap_control_mode", choices=["legacy", "repeatgrad_m2",
                                                         "late_m2_warmk4", "late_m1_warmk4",
                                                         "late_m1_sparsek4", "sparse_state_m2", "isotropic_sparse_m2",
                                                         "sampling_full_m1", "sampling_interleave_k2",
                                                         "sampling_interleave_k4",
                                                         "temporal_antithetic_m1",
                                                         "guidance_throughout_m1",
                                                         "temporal_antithetic_guidance_m1"],
                            default="legacy")
        parser.add_argument("--repeatgrad_validate", action="store_true")
        parser.add_argument("--dlap_guidance_grad_norm", choices=["raw", "normalized"],
                            default="raw")
        parser.add_argument("--dlap_guidance_refresh", type=int, default=1)
        parser.add_argument("--dlap_late_guidance_refresh", type=int, default=1)
        parser.add_argument("--dlap_state_refresh_interval", type=int, default=1)
        parser.add_argument("--dlap_param_grad_samples_per_refresh", type=int, default=2)
        parser.add_argument("--sparse_state_validate", action="store_true")
        parser.add_argument("--dlap_restore_validate", action="store_true")
        parser.add_argument("--p02_diagnostic", action="store_true",
                            help="Read-only P0-2 mechanism profiling and saved-state probes.")
        parser.add_argument("--p03_mode", choices=["none", "clean_m2", "fixed_early25",
                                                   "adaptive_t2_25"], default="none")
        parser.add_argument("--p03_calibration_json", type=str, default="")
        parser.add_argument("--tsnr_mode", choices=["none", "probe", "tsnr_p14", "temporal_ema"],
                            default="none",
                            help="M=1 Temporal-SNR study; probe leaves the FTM update unchanged.")
        parser.add_argument("--tsnr_beta", type=float, default=0.8)
        parser.add_argument("--tsnr_grid_size", type=int, default=14)
        parser.add_argument("--p06_decomp_probe", action="store_true",
                            help="Read-only P0-6 temporal directional decomposition probe; actual FTM-M1 update is unchanged.")
        parser.add_argument("--p06_beta", type=float, default=0.8)
        parser.add_argument("--p06_perp_scale", type=float, default=0.5)
        parser.add_argument("--use_spw", action="store_true",
                            help="P0-7A Stateful Parameter Walk M=1 attack.")
        parser.add_argument("--spw_step_size", type=float, default=0.05 / 299.0)
        parser.add_argument("--spw_radius", type=float, default=0.05)
        parser.add_argument("--csr_mode", choices=["none", "feature_only", "cross_space"],
                            default="none",
                            help="P0-8 current-step Cross-Space Reliability gate.")
        parser.add_argument("--csr_tau", type=float, default=1.0,
                            help="Frozen P0-8 layer-weight temperature; first-stage gate requires 1.0.")
        parser.add_argument("--lap_rho", type=float, default=0.05)
        parser.add_argument("--lap_rho_min", type=float, default=0.02)
        parser.add_argument(
            "--lap_layer_selector",
            type=str,
            default=None,
            choices=[None, "high", "mlp"],
        )
        parser.add_argument("--lap_num_samples", type=int, default=2)
        parser.add_argument("--lap_num_samples_interval", type=int, default=1)
        parser.add_argument("--lap_grad_smooth_kernel", type=int, default=0)
        parser.add_argument("--use_lap_adaptive_radius", action="store_true")
        parser.add_argument("--lap_beta", type=float, default=0.9)
        parser.add_argument("--lap_warmup", type=int, default=5)
        parser.add_argument("--lap_cv_temp", type=float, default=2.0)
        parser.add_argument(
            "--lap_mode",
            type=str,
            default="cv",
            choices=["cv", "consistency"],
        )
        parser.add_argument("--dlap_alpha", type=float, default=1.0)
        parser.add_argument("--dlap_disable_annealing", action="store_true")

    return parser


if __name__ == "__main__":
    main(argument_parsing().parse_args())
    print("DONE")
