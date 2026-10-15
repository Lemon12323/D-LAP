"""P0-8 current-step cross-space reliability for RTMF/FTM.

This module keeps the surrogate checkpoint fixed. It uses the *same* backward
call to obtain image gradients, FTM tuning gradients, and source-parameter
gradients for the currently active FTM layers. Parameter gradients are read-
only diagnostics: they never perturb or update source-model parameters.

Two frozen controls are provided:
  - feature_only: layer weights depend only on relative FTM tuning-gradient
    sensitivity.
  - cross_space: the same feature signal is discounted by same-layer relative
    parameter sensitivity.

Weights are normalized to mean 1 over the active layers of each surrogate so
that the average FTM tuning-update scale is preserved.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from attacks import FeatureTuning, LogitLoss, apply_input_transform, gkern, model_loss_sum


_EPS = 1e-12


def _count(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def _add_float(stats, key, value):
    if stats is not None:
        stats[key] = float(stats.get(key, 0.0)) + float(value)


def _combined_relative_param_sensitivity(
    params: List[torch.nn.Parameter], grads: List[Optional[torch.Tensor]]
) -> torch.Tensor:
    """Return ||g_theta||_2 / (||theta||_2 + eps) for one FTM layer module."""
    if not params:
        # This should not occur for the frozen Conv/Linear FTM configuration,
        # but a finite zero keeps the diagnostic path well-defined.
        device = grads[0].device if grads and grads[0] is not None else "cpu"
        return torch.zeros((), device=device)

    device = params[0].device
    dtype = params[0].dtype
    grad_sq = torch.zeros((), device=device, dtype=dtype)
    param_sq = torch.zeros((), device=device, dtype=dtype)
    for param, grad in zip(params, grads):
        param_sq = param_sq + param.detach().pow(2).sum()
        if grad is not None:
            grad_sq = grad.detach().pow(2).sum() + grad_sq
    return torch.sqrt(grad_sq) / (torch.sqrt(param_sq) + _EPS)


def _normalized_entropy(weights: torch.Tensor) -> float:
    """Entropy of mean-one weights, normalized to [0, 1] when n > 1."""
    n = int(weights.numel())
    if n <= 1:
        return 1.0
    probs = weights / weights.sum()
    entropy = -(probs * torch.log(probs + _EPS)).sum()
    return float((entropy / math.log(n)).item())


def cross_space_reliability_attack(
    source_models,
    x,
    target_label,
    exp_settings,
    device="cuda:0",
    mode="cross_space",
    tau=1.0,
    attack_type="RTMF",
    num_iter=300,
    max_epsilon=16,
    mu=1.0,
    compute_stats=None,
    memory_checkpoint=None,
):
    """Run the frozen P0-8 M=1 feature-only or cross-space reliability attack."""
    if mode not in ("feature_only", "cross_space"):
        raise ValueError("P0-8 mode must be 'feature_only' or 'cross_space'")
    if attack_type != "RTMF":
        raise ValueError("P0-8 is frozen to RTMF")
    if abs(float(tau) - 1.0) > 1e-12:
        raise ValueError("P0-8 first-stage gate freezes csr_tau=1.0")
    if len(source_models) != 1:
        raise ValueError("P0-8 first-stage gate requires one surrogate")

    for source_model in source_models:
        source_model.eval()

    # A private settings copy enables raw-feature-norm capture only inside the
    # P0-8 wrapper. The baseline FTM path is unchanged.
    local_settings = dict(exp_settings)
    local_settings["csr_track_feature_norms"] = True

    ti_kernel_size = 5
    kernel = gkern(ti_kernel_size, 3).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)

    eps = max_epsilon / 255.0
    alpha = local_settings["alpha"] / 255.0
    x_min = torch.clamp(x - eps, 0.0, 1.0)
    x_max = torch.clamp(x + eps, 0.0, 1.0)
    x_adv = x.clone()
    momentum = 0
    loss_fn = LogitLoss(target_label, local_settings["targeted"])

    img_width = x.size(-1)
    models = [FeatureTuning(model, img_width, local_settings, device) for model in source_models]

    # Record clean features exactly once, matching the ordinary FTM lifecycle.
    with torch.no_grad():
        for model in models:
            model.start_feature_record()
            model(x)
            _count(compute_stats, "source_model_forward_executions")
            _count(compute_stats, "clean_feature_cache_forwards")
            model.end_feature_record()

    if memory_checkpoint is not None:
        memory_checkpoint("after_feature_cache")

    try:
        # FTM consumes one nominal iteration for the clean-feature cache.
        for effective_step in range(1, num_iter):
            gradients_before = (
                compute_stats.get("image_gradient_evaluations", 0)
                if compute_stats is not None else 0
            )
            x_nes = x_adv.detach().requires_grad_(True)
            x_step = apply_input_transform(x_nes, attack_type, local_settings["p"])
            total_loss = model_loss_sum(models, x_step, loss_fn, compute_stats)

            # Gather current active FTM tuning leaves. For every active layer,
            # gather the parameters of the exact Conv/Linear module to which
            # the FeatureTuning hook is attached.
            tuning_records: List[Tuple[int, int, torch.Tensor]] = []
            param_records: Dict[Tuple[int, int], List[torch.nn.Parameter]] = {}
            param_targets: List[torch.nn.Parameter] = []
            param_slices: Dict[Tuple[int, int], Tuple[int, int]] = {}

            for model_idx, model in enumerate(models):
                active_indices = [
                    layer_idx
                    for layer_idx, triggered in model.mixing_triggered.items()
                    if triggered
                ]
                for layer_idx in active_indices:
                    tuning_records.append(
                        (model_idx, layer_idx, model.outputs_tuning[layer_idx])
                    )
                    module = model.layer_modules[layer_idx]
                    layer_params = [
                        p for p in module.parameters(recurse=False) if p.requires_grad
                    ]
                    start = len(param_targets)
                    param_targets.extend(layer_params)
                    param_slices[(model_idx, layer_idx)] = (start, len(param_targets))
                    param_records[(model_idx, layer_idx)] = layer_params

            grad_targets = [x_nes]
            grad_targets.extend(record[2] for record in tuning_records)
            grad_targets.extend(param_targets)

            all_grads = torch.autograd.grad(
                total_loss,
                grad_targets,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            _count(compute_stats, "backward_calls")
            _count(compute_stats, "image_gradient_evaluations")
            _count(compute_stats, "parameter_gradient_evaluations")
            _count(compute_stats, "csr_param_grad_requests")
            _count(compute_stats, "csr_steps")
            if mode == "feature_only":
                _count(compute_stats, "csr_feature_only_steps")
            else:
                _count(compute_stats, "csr_cross_space_steps")

            grad_x = all_grads[0]
            tuning_grad_offset = 1
            parameter_grad_offset = 1 + len(tuning_records)
            tuning_grads = all_grads[tuning_grad_offset:parameter_grad_offset]
            parameter_grads = all_grads[parameter_grad_offset:]

            # Group active layers by surrogate. The first-stage experiment uses
            # one surrogate, but keeping model-local normalization makes the
            # definition unambiguous for future audits.
            per_model_records: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]] = {}
            for record_index, (model_idx, layer_idx, _) in enumerate(tuning_records):
                feature_grad = tuning_grads[record_index]
                if feature_grad is None:
                    raise RuntimeError("Active FTM tuning tensor produced no gradient")

                feature_norm = models[model_idx].current_feature_norms.get(layer_idx)
                if feature_norm is None:
                    raise RuntimeError(
                        f"Missing current feature norm for active FTM layer {layer_idx}"
                    )
                feature_sensitivity = (
                    torch.linalg.vector_norm(feature_grad.detach())
                    / (feature_norm + _EPS)
                )

                start, end = param_slices[(model_idx, layer_idx)]
                layer_param_grads = list(parameter_grads[start:end])
                param_sensitivity = _combined_relative_param_sensitivity(
                    param_records[(model_idx, layer_idx)], layer_param_grads
                )

                if mode == "feature_only":
                    score = torch.log(feature_sensitivity + _EPS)
                else:
                    score = (
                        torch.log(feature_sensitivity + _EPS)
                        - torch.log(param_sensitivity + _EPS)
                    )

                per_model_records.setdefault(model_idx, []).append(
                    (record_index, score.detach(), param_sensitivity.detach())
                )
                _add_float(
                    compute_stats,
                    "csr_feature_sensitivity_sum",
                    feature_sensitivity.detach().item(),
                )
                _add_float(
                    compute_stats,
                    "csr_param_sensitivity_sum",
                    param_sensitivity.detach().item(),
                )
                _count(compute_stats, "csr_active_layers")

            weights_by_record: Dict[int, torch.Tensor] = {}
            for model_idx, records in per_model_records.items():
                scores = torch.stack([item[1] for item in records])
                # Stable softmax followed by mean-one normalization.
                weights = torch.softmax((scores - scores.max()) / tau, dim=0)
                weights = weights * len(records)
                _count(compute_stats, "csr_weighted_model_steps")
                if len(records) == 1:
                    _count(compute_stats, "csr_single_active_model_steps")
                _add_float(compute_stats, "csr_weight_max_sum", weights.max().item())
                _add_float(compute_stats, "csr_weight_entropy_sum", _normalized_entropy(weights))
                _add_float(
                    compute_stats,
                    "csr_score_std_sum",
                    scores.float().std(unbiased=False).item(),
                )
                for (record_index, _, _), weight in zip(records, weights):
                    weights_by_record[record_index] = weight.detach()

            # Apply the same FTM state update as the baseline, except for the
            # frozen mean-one reliability weight. No source parameter changes.
            for record_index, (model_idx, layer_idx, _) in enumerate(tuning_records):
                weight = weights_by_record.get(record_index)
                if weight is None:
                    weight = torch.ones((), device=tuning_grads[record_index].device)
                model = models[model_idx]
                model.outputs_tuning[layer_idx] = (
                    model.outputs_tuning[layer_idx]
                    - weight * tuning_grads[record_index]
                ).detach().requires_grad_(True)
            _count(compute_stats, "ftm_final_sample_state_update_steps")
            _count(compute_stats, "ftm_tuning_tensor_updates", len(tuning_records))

            # Standard RTMF image-gradient processing remains unchanged.
            grad_x = F.conv2d(
                grad_x,
                gaussian_kernel,
                bias=None,
                stride=1,
                padding=((ti_kernel_size - 1) // 2, (ti_kernel_size - 1) // 2),
                groups=3,
            )
            momentum = (
                mu * momentum
                + grad_x / torch.sum(torch.abs(grad_x), dim=[1, 2, 3], keepdim=True)
            )
            x_adv = torch.clamp(x_adv.detach() + alpha * momentum.sign(), x_min, x_max)
            _count(compute_stats, "adversarial_image_updates")
            _count(compute_stats, "effective_attack_steps")
            _count(compute_stats, "m1_image_update_steps")
            if compute_stats is not None:
                compute_stats.setdefault("samples_per_step", []).append(
                    compute_stats.get("image_gradient_evaluations", 0) - gradients_before
                )

            if memory_checkpoint is not None and effective_step in (1, 10, 50, num_iter - 1):
                memory_checkpoint("after_attack_step", effective_step)

            del (
                total_loss,
                x_step,
                grad_targets,
                all_grads,
                tuning_grads,
                parameter_grads,
                param_targets,
                param_records,
                param_slices,
                tuning_records,
                per_model_records,
                weights_by_record,
            )
    finally:
        for model in models:
            model.remove_hooks()

    return x_adv.detach()
