"""Project-wide CUDA memory policy and scalar-only analysis helpers.

Telemetry never changes tensor values, model state, RNG state, or batch size.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch


GIB = 1024 ** 3


def bytes_and_gib(value):
    value = int(value)
    return {"bytes": value, "gib": value / GIB}


@dataclass(frozen=True)
class MemoryPolicy:
    budget_gb: float = 10.0
    warn_gb: float = 9.5
    strict: bool = False
    monitor: bool = False

    def __post_init__(self):
        if self.budget_gb <= 0 or self.warn_gb <= 0 or self.warn_gb >= self.budget_gb:
            raise ValueError("Require 0 < vram_warn_gb < vram_budget_gb")

    def state(self, peak_bytes):
        peak_gib = int(peak_bytes) / GIB
        if peak_gib >= self.budget_gb:
            return "unsafe"
        if peak_gib >= self.warn_gb:
            return "warning"
        return "safe"

    def metadata(self):
        return {"budget_gib": self.budget_gb, "warn_gib": self.warn_gb,
                "strict": self.strict, "monitor": self.monitor,
                "amp_enabled": False, "generation_auto_resize_enabled": False}


def generation_signature(args, exp_settings, source_weights, gpu_name):
    """Hash generation-relevant settings, excluding paths and image count.

    A preflight and full run can use different manifests but must use the same
    batch size, source weights, model, attack settings, and GPU.
    """
    ignored = {"save_dir", "sample_manifest_input", "experiment_condition",
               "num_images", "eval_batch_size", "vram_budget_gb", "vram_warn_gb",
               "vram_monitor", "vram_strict", "vram_preflight",
               "vram_preflight_report", "target_model_names", "use_local_vit_targets",
               "debug"}
    settings = {k: v for k, v in vars(args).items() if k not in ignored}
    configuration = {k: v for k, v in exp_settings.items()
                     if k not in ("num_images", "target_model_names")}
    root = Path(__file__).resolve().parent
    code_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("main.py", "attacks.py", "dlap_attack.py",
                                "utils.py", "config.py", "memory_utils.py")}
    if getattr(args, "p02_diagnostic", False):
        code_hashes["p02_diagnostics.py"] = hashlib.sha256(
            (root / "p02_diagnostics.py").read_bytes()).hexdigest()
    if getattr(args, "p03_mode", "none") != "none":
        code_hashes["clean_repeatgrad.py"] = hashlib.sha256(
            (root / "clean_repeatgrad.py").read_bytes()).hexdigest()
        if args.p03_mode == "adaptive_t2_25":
            code_hashes["adaptive_trigger_calibration.json"] = hashlib.sha256(
                Path(args.p03_calibration_json).read_bytes()).hexdigest()
    if getattr(args, "tsnr_mode", "none") != "none":
        code_hashes["tsnr_attack.py"] = hashlib.sha256(
            (root / "tsnr_attack.py").read_bytes()).hexdigest()
    if getattr(args, "p06_decomp_probe", False):
        code_hashes["temporal_decomp.py"] = hashlib.sha256(
            (root / "temporal_decomp.py").read_bytes()).hexdigest()
    if getattr(args, "use_spw", False):
        code_hashes["stateful_param_walk.py"] = hashlib.sha256(
            (root / "stateful_param_walk.py").read_bytes()).hexdigest()
    if getattr(args, "csr_mode", "none") != "none":
        code_hashes["cross_space_reliability.py"] = hashlib.sha256(
            (root / "cross_space_reliability.py").read_bytes()).hexdigest()
    payload = {"args": settings, "exp_settings": configuration,
               "source_weights": source_weights, "gpu": gpu_name,
               "code_sha256": code_hashes, "pytorch_version": torch.__version__,
               "cuda_version": torch.version.cuda}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def likely_memory_consumers(attack_type, use_dlap=False, ensemble_size=1, use_spw=False, use_csr=False):
    categories = ["surrogate model parameters", "batch images and image gradients",
                  "forward/backward autograd intermediates"]
    if "F" in attack_type or "C" in attack_type:
        categories.append("cached clean features and feature-tuning state")
    if use_dlap:
        categories.append("parameter gradients, guidance EMA and perturbation directions")
    if use_spw:
        categories.append("same-backward parameter gradients and persistent checkpoint anchors")
    if use_csr:
        categories.append("same-backward active-layer parameter gradients and scalar reliability statistics")
    if ensemble_size > 1:
        categories.append("ensemble surrogate copies")
    return categories


def validate_preflight(report_path, expected_signature, policy, batch_size,
                       free_at_start_bytes=None):
    with open(report_path, encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("run_kind") != "preflight":
        raise ValueError("VRAM report is not a completed preflight")
    if report.get("generation_signature_sha256") != expected_signature:
        raise ValueError("Preflight configuration differs from this generation run")
    if report.get("generation_batch_size") != batch_size:
        raise ValueError("Preflight generation batch size differs")
    if report.get("images_processed") != batch_size:
        raise ValueError("Preflight must execute one full generation batch")
    peak = report["peak_allocated"]["bytes"]
    if (policy.strict and free_at_start_bytes is not None
            and free_at_start_bytes < peak + int(0.5 * GIB)):
        raise RuntimeError(
            "Strict VRAM preflight cannot be reused with current GPU occupancy: "
            "free {:.3f} GiB, preflight peak allocated {:.3f} GiB, required "
            "headroom 0.5 GiB. No batch size was changed.".format(
                free_at_start_bytes / GIB, peak / GIB))
    if policy.strict and policy.state(peak) == "unsafe":
        raise RuntimeError(
            "Strict VRAM preflight failed: model {}, attack {}, peak allocated "
            "{:.3f} GiB, peak reserved {:.3f} GiB, batch {}. Likely consumers: {}. "
            "Rerun explicitly with a lower candidate batch size; no automatic "
            "resizing was applied.".format(
                report.get("source_model_name", "unknown"), report.get("attack_type", "unknown"),
                peak / GIB, report["peak_reserved"]["gib"], batch_size,
                ", ".join(report.get("likely_consumers", []))))
    return report


class CudaMemoryTracker:
    """One experiment-wide peak reset, bounded optional step snapshots."""

    def __init__(self, device, policy):
        self.device = device
        self.policy = policy
        self.enabled = str(device).startswith("cuda") and torch.cuda.is_available()
        self.snapshots = []
        self.free_at_start_bytes = None
        self.total_bytes = (torch.cuda.get_device_properties(device).total_memory
                            if self.enabled else 0)
        self.gpu_name = torch.cuda.get_device_name(device) if self.enabled else None

    def start(self):
        if self.enabled:
            self.free_at_start_bytes = torch.cuda.mem_get_info(self.device)[0]
            torch.cuda.reset_peak_memory_stats(self.device)
        self.snapshot("experiment_start")

    def snapshot(self, name, batch_index=None, step=None):
        allocated = torch.cuda.memory_allocated(self.device) if self.enabled else 0
        reserved = torch.cuda.memory_reserved(self.device) if self.enabled else 0
        item = {"name": name, "allocated": bytes_and_gib(allocated),
                "reserved": bytes_and_gib(reserved)}
        if batch_index is not None:
            item["batch_index"] = int(batch_index)
        if step is not None:
            item["effective_step"] = int(step)
        self.snapshots.append(item)
        return item

    def attack_checkpoint(self, name, batch_index, step=None):
        if self.policy.monitor:
            self.snapshot(name, batch_index, step)

    def peak(self):
        if not self.enabled:
            return 0, 0
        return (torch.cuda.max_memory_allocated(self.device),
                torch.cuda.max_memory_reserved(self.device))

    def report(self, run_kind, signature, generation_batch_size,
               evaluation_batch_size, images_processed, preflight_report=None):
        allocated, reserved = self.peak()
        state = self.policy.state(allocated) if self.enabled else "unavailable"
        relevant = [s for s in self.snapshots if s["name"] in
                    ("before_attack", "after_attack", "after_model_cleanup")]
        before = next((s for s in relevant if s["name"] == "before_attack"), None)
        after = next((s for s in reversed(relevant) if s["name"] == "after_attack"), None)
        cleanup = next((s for s in reversed(relevant) if s["name"] == "after_model_cleanup"), None)
        step_checkpoints = [s for s in self.snapshots if s["name"] == "after_attack_step"]
        by_batch = {}
        for item in step_checkpoints:
            by_batch.setdefault(item["batch_index"], []).append(item)
        growth_candidates = []
        for batch, points in by_batch.items():
            late = [p for p in points if p["effective_step"] >= 10]
            allocations = [p["allocated"]["bytes"] for p in late]
            if (len(allocations) >= 2 and allocations[-1] - allocations[0] > 0.25 * GIB
                    and all(a <= b for a, b in zip(allocations, allocations[1:]))):
                growth_candidates.append(batch)
        batch_baselines = [s["allocated"]["bytes"] for s in self.snapshots
                           if s["name"] == "before_attack"]
        across_batch_growth = (len(batch_baselines) >= 3
                               and batch_baselines[-1] - batch_baselines[0] > 0.25 * GIB
                               and all(a <= b for a, b in zip(batch_baselines, batch_baselines[1:])))
        return {
            "run_kind": run_kind, "gpu_model": self.gpu_name,
            "total_vram": bytes_and_gib(self.total_bytes),
            "free_vram_at_start": (bytes_and_gib(self.free_at_start_bytes)
                                   if self.free_at_start_bytes is not None else None),
            "generation_batch_size": generation_batch_size,
            "evaluation_batch_size": evaluation_batch_size,
            "images_processed": images_processed,
            "generation_signature_sha256": signature,
            "preflight_report": preflight_report,
            "policy": self.policy.metadata(),
            "peak_allocated": bytes_and_gib(allocated),
            "peak_reserved": bytes_and_gib(reserved),
            "allocated_before_attack": before["allocated"] if before else None,
            "reserved_before_attack": before["reserved"] if before else None,
            "allocated_after_attack": after["allocated"] if after else None,
            "reserved_after_attack": after["reserved"] if after else None,
            "allocated_after_model_cleanup": cleanup["allocated"] if cleanup else None,
            "reserved_after_model_cleanup": cleanup["reserved"] if cleanup else None,
            "warning_state": state,
            "possible_step_growth_batches": growth_candidates,
            "possible_across_batch_growth": across_batch_growth,
            "growth_heuristic_threshold_gib": 0.25,
            "snapshots": self.snapshots,
        }


class OnlineScalarMoments:
    """Welford accumulator; stores only count, mean, and squared deviations."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, scalar):
        self.count += 1
        delta = float(scalar) - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (float(scalar) - self.mean)

    def result(self):
        return {"count": self.count, "mean": self.mean,
                "variance": self.m2 / (self.count - 1) if self.count > 1 else 0.0}


@torch.no_grad()
def gradient_pair_scalars(first, second):
    """Return scalar dot/norm/cosine statistics without retaining gradients."""
    dot = torch.sum(first * second).item()
    norm1_sq = torch.sum(first * first).item()
    norm2_sq = torch.sum(second * second).item()
    cosine = dot / ((norm1_sq * norm2_sq) ** 0.5) if norm1_sq and norm2_sq else 0.0
    return {"dot": dot, "first_norm_sq": norm1_sq,
            "second_norm_sq": norm2_sq, "cosine": cosine}
