"""Single-sample Temporal-SNR attack and read-only mechanism probe.

This module is intentionally isolated from the historical FTM / D-LAP code paths.
It keeps the frozen RTMF/FTM forward-backward sequence at M=1 and changes only
how the already-computed TI-processed image gradient is admitted to momentum.

Modes
-----
probe:
    Follow the original FTM-M1 update exactly, while computing hypothetical
    TSNR and temporal-EMA candidate updates and saving source directions at five
    frozen checkpoints for target-gradient alignment.
tsnr_p14:
    Use a 14x14 temporal-reliability map derived only from past M=1 gradients to
    reweight the current normalized gradient before the original momentum update.
temporal_ema:
    Lightweight control: use a conventional temporal EMA of the current
    normalized gradient before the original momentum update.

The TSNR state is causal: the reliability map used at step t is computed from
statistics ending at t-1.  The current gradient is written into the statistics
only after the current candidate increment has been formed.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attacks import (FeatureTuning, LogitLoss, _count_compute,
                     apply_input_transform, gkern, model_loss_sum)


MODES = ("probe", "tsnr_p14", "temporal_ema")
CHECKPOINTS = (60, 120, 178, 250, 298)
EPS = 1e-12


def _flat(x):
    return x.reshape(x.shape[0], -1)


def _l1_normalize(x):
    # Match the historical FTM reduction expression to preserve probe parity.
    return x / torch.sum(torch.abs(x), dim=[1, 2, 3], keepdim=True)


def _cos(a, b):
    aa, bb = _flat(a), _flat(b)
    return (aa * bb).sum(1) / (
        torch.linalg.vector_norm(aa, dim=1) * torch.linalg.vector_norm(bb, dim=1) + EPS
    )


def _candidate(incoming, increment, mu):
    if isinstance(incoming, int):
        return increment
    return mu * incoming + increment


class TemporalReliabilityState:
    """Causal first/second-moment state over normalized TI gradients."""

    def __init__(self, beta=0.8, grid_size=14):
        if not 0.0 < beta < 1.0:
            raise ValueError("TSNR beta must be in (0, 1)")
        if grid_size < 1:
            raise ValueError("TSNR grid size must be positive")
        self.beta = float(beta)
        self.grid_size = int(grid_size)
        self.ema = None
        self.ema_sq = None
        self.steps = 0

    def _past_moments(self):
        if self.steps == 0:
            return None, None
        correction = 1.0 - self.beta ** self.steps
        mean = self.ema / correction
        second = self.ema_sq / correction
        variance = torch.clamp(second - mean.square(), min=0.0)
        return mean, variance

    def ema_control_increment(self, current):
        """Conventional temporal-EMA control increment."""
        if self.steps == 0:
            return current
        mean, _ = self._past_moments()
        return _l1_normalize(self.beta * mean + (1.0 - self.beta) * current)

    def tsnr_increment(self, current):
        """Return (TSNR increment, spatial confidence) from past statistics only."""
        if self.steps < 2:
            confidence = torch.ones(
                (current.shape[0], 1, current.shape[-2], current.shape[-1]),
                dtype=current.dtype, device=current.device)
            return current, confidence

        mean, variance = self._past_moments()
        signal2 = F.adaptive_avg_pool2d(mean.square().mean(dim=1, keepdim=True),
                                        (self.grid_size, self.grid_size))
        noise2 = F.adaptive_avg_pool2d(variance.mean(dim=1, keepdim=True),
                                       (self.grid_size, self.grid_size))
        reliability = torch.sqrt(signal2 + EPS) / (torch.sqrt(noise2 + EPS) + EPS)
        confidence_patch = reliability / (1.0 + reliability)
        confidence = F.interpolate(confidence_patch, size=current.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return _l1_normalize(current * confidence), confidence

    def increments(self, current):
        """Return both increments for read-only mechanism probing."""
        tsnr, confidence = self.tsnr_increment(current)
        ema = self.ema_control_increment(current)
        return tsnr, ema, confidence

    def update(self, current):
        value = current.detach()
        if self.ema is None:
            self.ema = torch.zeros_like(value)
            self.ema_sq = torch.zeros_like(value)
        self.ema.mul_(self.beta).add_(value, alpha=1.0 - self.beta)
        self.ema_sq.mul_(self.beta).addcmul_(value, value, value=1.0 - self.beta)
        self.steps += 1


def _save_probe_checkpoint(save_dir, batch_index, effective_step, x_adv, target_labels,
                           baseline_candidate, tsnr_candidate, ema_candidate,
                           beta, grid_size, confidence):
    root = Path(save_dir) / "tsnr_probe"
    root.mkdir(parents=True, exist_ok=True)
    stem = root / f"batch{batch_index}_step{effective_step}"
    torch.save({
        "x": x_adv.detach().cpu(),
        "target_labels": target_labels.detach().cpu(),
        "batch": int(batch_index),
        "effective_step": int(effective_step),
        "candidate_semantics": "incoming momentum plus current normalized increment at pre-update x_t",
    }, str(stem) + "_state.pt")
    np.savez(
        str(stem) + "_source.npz",
        M1=baseline_candidate.detach().cpu().numpy().astype(np.float32),
        TSNR=tsnr_candidate.detach().cpu().numpy().astype(np.float32),
        EMA=ema_candidate.detach().cpu().numpy().astype(np.float32),
    )
    meta = {
        "batch": int(batch_index),
        "effective_step": int(effective_step),
        "beta": float(beta),
        "grid_size": int(grid_size),
        "confidence_mean": float(confidence.mean().item()),
        "confidence_std": float(confidence.std(unbiased=False).item()),
        "confidence_min": float(confidence.min().item()),
        "confidence_max": float(confidence.max().item()),
    }
    Path(str(stem) + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def tsnr_ftm_attack(source_models, x, target_label, exp_settings, device,
                    mode="probe", attack_type="RTMF", num_iter=300,
                    max_epsilon=16, mu=1.0, beta=0.8, grid_size=14,
                    compute_stats=None, memory_checkpoint=None,
                    save_dir=None, batch_index=0):
    """Frozen M=1 RTMF/FTM attack with optional Temporal-SNR gradient gating."""
    if mode not in MODES:
        raise ValueError(f"Unknown TSNR mode: {mode}")
    if attack_type != "RTMF":
        raise ValueError("First-stage TSNR study is frozen to RTMF")
    if len(source_models) != 1 or not exp_settings.get("targeted", False):
        raise ValueError("First-stage TSNR study requires one targeted source model")
    if num_iter < 2 or mu != 1.0:
        raise ValueError("First-stage TSNR study requires >=2 nominal steps and momentum 1")

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

    # Match the historical FTM clean-feature cache exactly.
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

    state = TemporalReliabilityState(beta=beta, grid_size=grid_size)
    rows = []

    try:
        # FTM consumed one nominal iteration for the clean cache, leaving 299 updates.
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

            # Frozen RTMF contains TI, so process exactly before temporal logic.
            processed = F.conv2d(grad_x, gaussian_kernel, bias=None, stride=1,
                                 padding=2, groups=3)
            current = _l1_normalize(processed)
            incoming = momentum

            effective_step = step_index
            if mode == "probe":
                with torch.no_grad():
                    tsnr_increment, ema_increment, confidence = state.increments(current.detach())
                    baseline_candidate = _candidate(incoming, current.detach(), mu)
                    tsnr_candidate = _candidate(incoming, tsnr_increment, mu)
                    ema_candidate = _candidate(incoming, ema_increment, mu)
                    candidate_cos = _cos(baseline_candidate, tsnr_candidate)
                    candidate_sign = (_flat(baseline_candidate.sign()) !=
                                      _flat(tsnr_candidate.sign())).float().mean(1)
                    increment_cos = _cos(current.detach(), tsnr_increment)
                    ema_candidate_cos = _cos(baseline_candidate, ema_candidate)
                    conf_flat = confidence.reshape(confidence.shape[0], -1)
                    metrics = (candidate_cos, candidate_sign, increment_cos, ema_candidate_cos,
                               conf_flat.mean(1), conf_flat.std(1, unbiased=False),
                               conf_flat.min(1).values, conf_flat.max(1).values)

                if effective_step in CHECKPOINTS:
                    if save_dir is None:
                        raise ValueError("TSNR probe requires save_dir")
                    _save_probe_checkpoint(
                        save_dir, batch_index, effective_step, x_adv, target_label,
                        baseline_candidate, tsnr_candidate, ema_candidate,
                        beta, grid_size, confidence,
                    )
                    _count_compute(compute_stats, "tsnr_probe_checkpoints")

                arrays = [v.detach().cpu().numpy().tolist() for v in metrics]
                cols = ("baseline_tsnr_candidate_cosine", "baseline_tsnr_sign_disagreement",
                        "current_tsnr_increment_cosine", "baseline_ema_candidate_cosine",
                        "confidence_mean", "confidence_std", "confidence_min", "confidence_max")
                for i in range(x.shape[0]):
                    row = {"batch": batch_index, "image_index": batch_index * 32 + i,
                           "effective_step": effective_step, "mode": mode,
                           "history_steps": state.steps, "beta": beta, "grid_size": grid_size}
                    row.update({col: arrays[j][i] for j, col in enumerate(cols)})
                    rows.append(row)
                increment = current
            elif mode == "tsnr_p14":
                with torch.no_grad():
                    increment, _ = state.tsnr_increment(current.detach())
                _count_compute(compute_stats, "tsnr_filtered_updates")
            else:
                with torch.no_grad():
                    increment = state.ema_control_increment(current.detach())
                _count_compute(compute_stats, "temporal_ema_updates")

            # Causality: current observation enters the history only after the
            # reliability/candidate computation for this same step.
            with torch.no_grad():
                state.update(current.detach())
            _count_compute(compute_stats, "tsnr_reliability_steps")

            momentum = mu * momentum + increment
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

            del grad_x, processed, current, increment
            if mode == "probe":
                del tsnr_increment, ema_increment, confidence
                del baseline_candidate, tsnr_candidate, ema_candidate
    finally:
        for model in models:
            model.remove_hooks()

    if save_dir is not None and mode == "probe":
        path = Path(save_dir) / "tsnr_profile" / f"batch{batch_index}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    return x_adv.detach()
