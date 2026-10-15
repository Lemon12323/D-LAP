"""P0-3 clean RepeatGrad and frozen whole-batch sample-2 schedules.

The historical RepeatGrad control remains in dlap_attack.py.  This module
keeps its feature-tuning and random-draw order while omitting unused model
parameter gradients and all parameter-state computations.
"""
import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attacks import FeatureTuning, LogitLoss, RDI, gkern, _count_compute
from dlap_attack import (_clear_cached_attention_biases, _collect_param_groups,
                         _sample_per_group_direction)


MODES = ("clean_m2", "fixed_early25", "adaptive_t2_25")
EPS = 1e-12


def _cos(a, b):
    aa, bb = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return (aa * bb).sum(1) / (aa.norm(dim=1) * bb.norm(dim=1) + EPS)


def clean_repeatgrad_attack(source_models, x, target_label, exp_settings, device,
                            mode, threshold=None, num_iter=300, max_epsilon=16,
                            mu=1.0, compute_stats=None, memory_checkpoint=None,
                            save_dir=None, batch_index=0, validate_parameters=False):
    if mode not in MODES or num_iter < 2 or mu != 1.0:
        raise ValueError("P0-3 requires a frozen mode, at least two nominal steps and momentum 1")
    if mode == "adaptive_t2_25" and threshold is None:
        raise ValueError("Adaptive mode requires the frozen P0-2 threshold")
    if len(source_models) != 1 or not exp_settings["targeted"]:
        raise ValueError("P0-3 uses the single targeted ResNet50 source")
    for source_model in source_models:
        source_model.eval()
    kernel = gkern(5, 3).astype(np.float32)
    gaussian_kernel = torch.from_numpy(np.expand_dims(np.stack([kernel] * 3), 1)).to(device)
    eps, alpha = max_epsilon / 255.0, exp_settings["alpha"] / 255.0
    x_min, x_max = torch.clamp(x - eps, 0, 1), torch.clamp(x + eps, 0, 1)
    x_adv = x.clone()
    momentum = 0
    loss_fn = LogitLoss(target_label, True)
    with torch.no_grad():
        models = [FeatureTuning(sm, x.size(-1), exp_settings, device) for sm in source_models]
        for model in models:
            model.start_feature_record()
            model(x)
            _count_compute(compute_stats, "source_model_forward_executions")
            _count_compute(compute_stats, "clean_feature_cache_forwards")
            model.end_feature_record()
    if memory_checkpoint is not None:
        memory_checkpoint("after_feature_cache")
    # This list is used only to reproduce historical random draws. No selected
    # parameter gradient, EMA, radius or perturbation is ever constructed.
    direction_params = []
    direction_group_ids = []
    direction_groups = 0
    for sm in source_models:
        params, gids, groups = _collect_param_groups(sm)
        direction_params.extend(params)
        direction_group_ids.extend(gid + direction_groups for gid in gids)
        direction_groups += groups
    versions = [p._version for p in direction_params] if validate_parameters else None
    originals = [p.detach().clone() for p in direction_params] if validate_parameters else None
    rows = []

    def sample(x_current):
        direction = _sample_per_group_direction(
            direction_params, direction_group_ids, direction_groups)
        _count_compute(compute_stats, "random_direction_draw_calls")
        _count_compute(compute_stats, "random_direction_tensor_draw_calls", len(direction_params))
        del direction
        x_nes = x_current.detach().requires_grad_(True)
        x_in = RDI(x_nes)
        _count_compute(compute_stats, "rtmf_transform_calls")
        _count_compute(compute_stats, "rtmf_transform_random_scalar_draws", 3)
        total_loss = 0
        _count_compute(compute_stats, "logical_attack_forward_calls")
        for model in models:
            _clear_cached_attention_biases(model)
            logits = model(x_in)
            total_loss += loss_fn(logits)
            _count_compute(compute_stats, "source_model_forward_executions")
            decisions = len(model.mixing_triggered)
            triggered = sum(bool(v) for v in model.mixing_triggered.values())
            _count_compute(compute_stats, "ftm_mixing_decision_draw_calls", decisions)
            _count_compute(compute_stats, "ftm_shuffle_draw_calls",
                           triggered if exp_settings["shuffle_image_feature"] == "SelfShuffle" else 0)
            _count_compute(compute_stats, "ftm_coefficient_draw_calls", triggered)
            del logits
        grad_targets = [x_nes]
        active = []
        for model in models:
            indices = [li for li, used in model.mixing_triggered.items() if used]
            grad_targets.extend(model.outputs_tuning[li] for li in indices)
            active.append(indices)
        grads = torch.autograd.grad(total_loss, grad_targets, retain_graph=False,
                                    create_graph=False)
        _count_compute(compute_stats, "backward_calls")
        _count_compute(compute_stats, "image_gradient_evaluations")
        del grad_targets, x_in, total_loss
        return grads[0], grads[1:], active

    def apply_ftm_update(feature_grads, active):
        offset = 0
        for model, indices in zip(models, active):
            for i, layer in enumerate(indices):
                model.outputs_tuning[layer] = (
                    model.outputs_tuning[layer] - feature_grads[offset + i]
                ).detach().requires_grad_(True)
            offset += len(indices)
        _count_compute(compute_stats, "ftm_final_sample_state_update_steps")
        _count_compute(compute_stats, "ftm_tuning_tensor_updates", offset)

    try:
        for step in range(num_iter - 1):
            incoming = momentum
            g1, fg1, active1 = sample(x_adv)
            t2 = float(g1.detach().abs().sum(dim=(1, 2, 3)).mean().item())
            use_second = (mode == "clean_m2" or
                          (mode == "fixed_early25" and step < 75) or
                          (mode == "adaptive_t2_25" and t2 <= threshold))
            # The first-sample backward has completed before this decision.
            if use_second:
                g2, fg2, active2 = sample(x_adv)
                apply_ftm_update(fg2, active2)
                grad_x = (g1 + g2) / 2
                _count_compute(compute_stats, "m2_image_update_steps")
                _count_compute(compute_stats, "sample2_trigger_count")
                with torch.no_grad():
                    m1 = F.conv2d(g1, gaussian_kernel, padding=2, groups=3)
                    m2 = F.conv2d(grad_x, gaussian_kernel, padding=2, groups=3)
                    m1 = m1 / m1.abs().sum(dim=(1, 2, 3), keepdim=True)
                    m2 = m2 / m2.abs().sum(dim=(1, 2, 3), keepdim=True)
                    if not isinstance(incoming, int):
                        m1 = mu * incoming + m1
                        m2 = mu * incoming + m2
                    raw_cos = float(_cos(g1, g2).mean().item())
                    update_gap = float((1 - _cos(m1, m2)).mean().item())
                    sign_disagreement = float((m1.sign() != m2.sign()).float().mean().item())
                del g2, fg2, active2, m1, m2
            else:
                apply_ftm_update(fg1, active1)
                grad_x = g1
                _count_compute(compute_stats, "m1_image_update_steps")
                raw_cos = update_gap = sign_disagreement = ""
            rows.append({"batch": batch_index, "step": step, "t2_batch_mean_l1": t2,
                         "threshold": threshold if threshold is not None else "",
                         "use_sample2": int(use_second), "decision_after_sample1": 1,
                         "g1_g2_cosine": raw_cos, "update_gap": update_gap,
                         "sign_disagreement": sign_disagreement})
            del fg1, active1, g1
            grad_x = F.conv2d(grad_x, gaussian_kernel, padding=2, groups=3)
            momentum = mu * momentum + grad_x / grad_x.abs().sum(dim=(1, 2, 3), keepdim=True)
            x_adv = x_adv.detach() + alpha * momentum.sign()
            x_adv = torch.clamp(x_adv, x_min, x_max)
            _count_compute(compute_stats, "adversarial_image_updates")
            _count_compute(compute_stats, "effective_attack_steps")
            if compute_stats is not None:
                compute_stats.setdefault("samples_per_step", []).append(2 if use_second else 1)
            effective_step = step + 1
            if memory_checkpoint is not None and effective_step in (1, 10, 50, 299):
                memory_checkpoint("after_attack_step", effective_step)
    finally:
        for model in models:
            model.remove_hooks()
    if validate_parameters:
        if any(p._version != v or not torch.equal(p.detach(), original)
               for p, v, original in zip(direction_params, versions, originals)):
            raise AssertionError("Clean RepeatGrad modified a source parameter")
        compute_stats["repeatgrad_parameter_integrity_validated"] = True
    if compute_stats is not None:
        compute_stats["parameter_perturbation_linf_max"] = 0.0
        compute_stats["rng_state_end_sha256"] = {
            "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "torch_cuda": hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest(),
            "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
            "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        }
    if save_dir is not None:
        path = Path(save_dir) / "p03_steps" / f"batch{batch_index}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return x_adv.detach()
