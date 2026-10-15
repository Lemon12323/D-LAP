"""P0-6 read-only temporal directional decomposition probe.

The attack trajectory is the frozen ResNet50 RTMF/FTM M=1 baseline.  This
module never changes that trajectory.  It uses only already-computed normalized
TI gradients to maintain a causal short-horizon temporal reference and, at five
checkpoints, forms hypothetical candidate update directions:

M1
    Historical FTM candidate: incoming attack momentum + current increment.
EMA
    P0-5 temporal-EMA control, retained as a positive-control/reference.
PARALLEL
    Keep only the component of the current increment parallel to the past
    temporal reference, then L1-normalize it before adding incoming momentum.
PERP
    Keep only the temporal-orthogonal component of the current increment.
HALF
    Keep the full parallel component but scale the orthogonal component by the
    frozen factor 0.5, then L1-normalize before adding incoming momentum.

The temporal reference used at step t is based only on steps < t.  The current
observation is written into the EMA after all current-step probes are formed.
No additional surrogate forward/backward is performed.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attacks import (FeatureTuning, LogitLoss, _count_compute,
                     apply_input_transform, gkern, model_loss_sum)


CHECKPOINTS = (60, 120, 178, 250, 298)
ESTIMATORS = ("M1", "EMA", "PARALLEL", "PERP", "HALF")
EPS = 1e-12


def _flat(x):
    return x.reshape(x.shape[0], -1)


def _l1_normalize(x):
    # Match the historical FTM expression for the actual baseline path.
    return x / torch.sum(torch.abs(x), dim=[1, 2, 3], keepdim=True)


def _safe_l1_normalize(x):
    denom = torch.sum(torch.abs(x), dim=[1, 2, 3], keepdim=True)
    return x / (denom + EPS)


def _l2(x):
    return torch.linalg.vector_norm(_flat(x), dim=1)


def _cos(a, b):
    aa, bb = _flat(a), _flat(b)
    return (aa * bb).sum(1) / (
        torch.linalg.vector_norm(aa, dim=1) * torch.linalg.vector_norm(bb, dim=1) + EPS
    )


def _candidate(incoming, increment, mu):
    if isinstance(incoming, int):
        return increment
    return mu * incoming + increment


class TemporalDirectionState:
    """Causal EMA reference over normalized TI gradients."""

    def __init__(self, beta=0.8):
        if not 0.0 < beta < 1.0:
            raise ValueError("P0-6 temporal beta must be in (0, 1)")
        self.beta = float(beta)
        self.ema = None
        self.steps = 0

    def past_mean(self):
        if self.steps == 0:
            return None
        correction = 1.0 - self.beta ** self.steps
        return self.ema / correction

    def ema_control_increment(self, current):
        """Exactly mirror the P0-5 temporal-EMA control semantics."""
        mean = self.past_mean()
        if mean is None:
            return current
        return _safe_l1_normalize(self.beta * mean + (1.0 - self.beta) * current)

    def decompose(self, current, perp_scale=0.5):
        """Decompose current into components parallel/orthogonal to past mean.

        The projection is performed independently per image over the complete
        attack-space gradient vector.  Scaling of the reference does not affect
        the projection geometry.
        """
        reference = self.past_mean()
        if reference is None:
            zero = torch.zeros_like(current)
            one = torch.ones(current.shape[0], dtype=current.dtype, device=current.device)
            metrics = {
                "current_reference_cosine": torch.full_like(one, float("nan")),
                "parallel_l2_fraction": torch.full_like(one, float("nan")),
                "perp_l2_fraction": torch.full_like(one, float("nan")),
                "projection_coefficient": torch.full_like(one, float("nan")),
                "anti_aligned": torch.zeros_like(one),
            }
            return current, current, current, reference, metrics

        cur = _flat(current)
        ref = _flat(reference)
        ref_sq = (ref * ref).sum(1) + EPS
        coefficient = (cur * ref).sum(1) / ref_sq
        parallel = coefficient[:, None, None, None] * reference
        perpendicular = current - parallel
        half = parallel + float(perp_scale) * perpendicular

        cur_norm = _l2(current) + EPS
        metrics = {
            "current_reference_cosine": _cos(current, reference),
            "parallel_l2_fraction": _l2(parallel) / cur_norm,
            "perp_l2_fraction": _l2(perpendicular) / cur_norm,
            "projection_coefficient": coefficient,
            "anti_aligned": (coefficient < 0).to(current.dtype),
        }
        return (_safe_l1_normalize(parallel),
                _safe_l1_normalize(perpendicular),
                _safe_l1_normalize(half),
                reference, metrics)

    def update(self, current):
        value = current.detach()
        if self.ema is None:
            self.ema = torch.zeros_like(value)
        self.ema.mul_(self.beta).add_(value, alpha=1.0 - self.beta)
        self.steps += 1


def _save_checkpoint(save_dir, batch_index, effective_step, x_adv, target_labels,
                     incoming, current, reference, beta, perp_scale, scalar_metrics):
    root = Path(save_dir) / "p06_probe"
    root.mkdir(parents=True, exist_ok=True)
    stem = root / f"batch{batch_index}_step{effective_step}"
    torch.save({
        "x": x_adv.detach().cpu(),
        "target_labels": target_labels.detach().cpu(),
        "batch": int(batch_index),
        "effective_step": int(effective_step),
        "candidate_semantics": (
            "candidate = incoming historical FTM momentum + hypothetical current increment; "
            "the actual trajectory always uses M1"
        ),
        "increment_semantics": "TI-processed per-image L1-normalized current-step direction",
        "reference_semantics": "bias-corrected EMA over normalized TI gradients from steps < t",
    }, str(stem) + "_state.pt", _use_new_zipfile_serialization=False)
    if reference is None:
        raise AssertionError("P0-6 checkpoint requires a non-empty causal reference")
    incoming_tensor = (torch.zeros_like(current)
                       if isinstance(incoming, int) else incoming.detach())
    # Persist only the three independent basis tensors. EMA / PARALLEL / PERP / HALF
    # are deterministically reconstructed during target analysis. This cuts source
    # checkpoint storage in half without changing the attack trajectory or geometry.
    current_np = current.detach().cpu().numpy().astype(np.float32)
    reference_np = reference.detach().cpu().numpy().astype(np.float32)
    incoming_np = incoming_tensor.cpu().numpy().astype(np.float32)

    np.save(str(stem) + "_current.npy", current_np, allow_pickle=False)
    np.save(str(stem) + "_reference.npy", reference_np, allow_pickle=False)
    np.save(str(stem) + "_incoming.npy", incoming_np, allow_pickle=False)
    meta = {
        "batch": int(batch_index),
        "effective_step": int(effective_step),
        "beta": float(beta),
        "perp_scale": float(perp_scale),
        "estimators": list(ESTIMATORS),
    }
    for key, value in scalar_metrics.items():
        if torch.is_tensor(value):
            finite = value[torch.isfinite(value)]
            meta[f"{key}_mean"] = (float(finite.mean().item()) if finite.numel() else None)
        else:
            meta[key] = value
    Path(str(stem) + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def temporal_decomp_probe_attack(source_models, x, target_label, exp_settings, device,
                                 attack_type="RTMF", num_iter=300,
                                 max_epsilon=16, mu=1.0, beta=0.8,
                                 perp_scale=0.5, compute_stats=None,
                                 memory_checkpoint=None, save_dir=None,
                                 batch_index=0, save_probe_artifacts=True):
    """Run frozen FTM-M1 while collecting read-only P0-6 directions."""
    if attack_type != "RTMF":
        raise ValueError("P0-6 is frozen to RTMF")
    if len(source_models) != 1 or not exp_settings.get("targeted", False):
        raise ValueError("P0-6 requires one targeted source model")
    if num_iter < 2 or mu != 1.0:
        raise ValueError("P0-6 requires >=2 nominal steps and momentum 1")
    if abs(float(perp_scale) - 0.5) > 1e-12:
        raise ValueError("P0-6 first screen freezes perpendicular scale at 0.5")

    for source_model in source_models:
        source_model.eval()

    prob = exp_settings["p"]
    lr = exp_settings["alpha"]
    kernel = gkern(5, 3).astype(np.float32)
    gaussian_kernel = torch.from_numpy(
        np.expand_dims(np.stack([kernel] * 3), 1)
    ).to(device)

    eps = max_epsilon / 255.0
    alpha = lr / 255.0
    x_min = torch.clamp(x - eps, 0.0, 1.0)
    x_max = torch.clamp(x + eps, 0.0, 1.0)
    x_adv = x.clone()
    momentum = 0
    loss_fn = LogitLoss(target_label, True)

    # Keep the historical RTMF/FTM clean-feature cache sequence unchanged.
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

    state = TemporalDirectionState(beta=beta)
    rows = []

    try:
        for step_index in range(1, num_iter):
            gradients_before_step = (compute_stats.get("image_gradient_evaluations", 0)
                                     if compute_stats is not None else 0)
            x_nes = x_adv.detach().requires_grad_(True)
            x_step = apply_input_transform(x_nes, attack_type, prob)
            total_loss = model_loss_sum(models, x_step, loss_fn, compute_stats)

            all_params = [x_nes]
            all_active_layers = []
            for model in models:
                tuning_params = []
                active_layer_indices = []
                for layer_idx, was_triggered in model.mixing_triggered.items():
                    if was_triggered:
                        tuning_params.append(model.outputs_tuning[layer_idx])
                        active_layer_indices.append(layer_idx)
                all_params.extend(tuning_params)
                all_active_layers.append((active_layer_indices, len(tuning_params)))

            all_grads = torch.autograd.grad(total_loss, all_params,
                                            retain_graph=False, create_graph=False)
            _count_compute(compute_stats, "backward_calls")
            _count_compute(compute_stats, "image_gradient_evaluations")
            grad_x = all_grads[0]

            current_idx = 1
            for model_idx, (active_indices, num_params) in enumerate(all_active_layers):
                model = models[model_idx]
                model_grads = all_grads[current_idx:current_idx + num_params]
                for layer_idx, grad in zip(active_indices, model_grads):
                    model.outputs_tuning[layer_idx] = (
                        model.outputs_tuning[layer_idx] - grad
                    ).detach().requires_grad_(True)
                current_idx += num_params

            del total_loss, x_step, all_grads, all_params, all_active_layers

            processed = F.conv2d(grad_x, gaussian_kernel, bias=None, stride=1,
                                 padding=2, groups=3)
            current = _l1_normalize(processed)
            incoming = momentum
            effective_step = step_index

            with torch.no_grad():
                parallel_inc, perp_inc, half_inc, reference, decomp = state.decompose(
                    current.detach(), perp_scale=perp_scale)
                ema_inc = state.ema_control_increment(current.detach())
                increments = {
                    "M1": current.detach(),
                    "EMA": ema_inc,
                    "PARALLEL": parallel_inc,
                    "PERP": perp_inc,
                    "HALF": half_inc,
                }
                candidates = {name: _candidate(incoming, increment, mu)
                              for name, increment in increments.items()}
                scalar_metrics = dict(decomp)
                scalar_metrics.update({
                    "m1_ema_candidate_cosine": _cos(candidates["M1"], candidates["EMA"]),
                    "m1_parallel_candidate_cosine": _cos(candidates["M1"], candidates["PARALLEL"]),
                    "m1_perp_candidate_cosine": _cos(candidates["M1"], candidates["PERP"]),
                    "m1_half_candidate_cosine": _cos(candidates["M1"], candidates["HALF"]),
                    "m1_half_sign_disagreement": (
                        _flat(candidates["M1"].sign()) != _flat(candidates["HALF"].sign())
                    ).float().mean(1),
                })

            if effective_step in CHECKPOINTS:
                if save_probe_artifacts:
                    if save_dir is None:
                        raise ValueError("P0-6 probe requires save_dir")
                    _save_checkpoint(
                        save_dir, batch_index, effective_step, x_adv, target_label,
                        incoming, current.detach(), reference, beta, perp_scale, scalar_metrics,
                    )
                _count_compute(compute_stats, "p06_probe_checkpoints")

            if save_probe_artifacts:
                arrays = [v.detach().cpu().numpy().tolist() for v in scalar_metrics.values()]
                cols = list(scalar_metrics)
                for i in range(x.shape[0]):
                    row = {
                        "batch": int(batch_index),
                        "image_index": int(batch_index * 32 + i),
                        "effective_step": int(effective_step),
                        "history_steps": int(state.steps),
                        "beta": float(beta),
                        "perp_scale": float(perp_scale),
                    }
                    row.update({col: arrays[j][i] for j, col in enumerate(cols)})
                    rows.append(row)

            # Causality: write current only after all step-t candidates are formed.
            with torch.no_grad():
                state.update(current.detach())
            _count_compute(compute_stats, "p06_decomp_steps")

            # Critical: the actual attack always follows the unmodified M1 path.
            momentum = mu * momentum + current
            x_adv = x_adv.detach() + alpha * momentum.sign()
            x_adv = torch.clamp(x_adv, x_min, x_max)
            _count_compute(compute_stats, "adversarial_image_updates")
            _count_compute(compute_stats, "effective_attack_steps")
            _count_compute(compute_stats, "m1_image_update_steps")
            if compute_stats is not None:
                compute_stats.setdefault("samples_per_step", []).append(
                    compute_stats.get("image_gradient_evaluations", 0) - gradients_before_step
                )

            if memory_checkpoint is not None and effective_step in (1, 10, 50, num_iter - 1):
                memory_checkpoint("after_attack_step", effective_step)

            del grad_x, processed, current, parallel_inc, perp_inc, half_inc, ema_inc
            del increments, candidates, scalar_metrics
    finally:
        for model in models:
            model.remove_hooks()

    if save_probe_artifacts:
        if save_dir is None:
            raise ValueError("P0-6 full probe requires save_dir")
        if not rows:
            raise AssertionError("P0-6 full probe produced no profile rows")
        path = Path(save_dir) / "p06_profile" / f"batch{batch_index}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    return x_adv.detach()
