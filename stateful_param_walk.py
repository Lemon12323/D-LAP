"""P0-7A Stateful Parameter Walk (SPW), single-sample M=1 attack.

This module is intentionally isolated from the historical FTM, D-LAP, TSNR,
and P0-6 code paths.  It keeps the frozen RTMF/FTM image-update pipeline at
one source forward/backward per effective step while making the surrogate
parameter state persistent across attack iterations.

At effective step t, the same autograd traversal returns the image gradient,
FTM tuning gradients, and selected source-parameter gradients at theta_t.
The image gradient updates x_t in the ordinary FTM/TI/momentum path.  The
parameter gradient then makes a target-adverse, group-normalized step

    theta_{t+1} = Proj_{B(theta_0, radius)}(
        theta_t - step_size * NormalizeGroup(grad_theta_t)
    )

so theta_{t+1} first influences the image gradient at the next attack step.
The walk is projected into a per-module-group L2 ball around the original
checkpoint state and the selected parameters are restored exactly at the end
of every image batch (including exceptional exits).

First-stage-v2 constraints are enforced again in main.py: ResNet50 source,
RTMF, batch 32 (or five-image smoke), 300 nominal iterations, epsilon 16/255,
MI momentum 1, one source model, radius=0.05 and a horizon-calibrated step
size radius/(299 effective updates).  Normalization affine parameters are
excluded from the persistent walk by module type.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks import (
    FeatureTuning,
    LogitLoss,
    _count_compute,
    apply_input_transform,
    gkern,
    model_loss_sum,
)
from dlap_attack import _normalized_gradient_per_group


EPS = 1e-12


def _collect_spw_param_groups(model):
    """Collect persistent-walk parameters while excluding normalization state.

    D-LAP's historical name-based selector does not reliably exclude torchvision
    BatchNorm modules (e.g. ``bn1`` and ``downsample.1``).  That is tolerable for
    temporary perturbations but unsafe for a persistent 299-step walk.  SPW uses
    module types so BatchNorm/LayerNorm/GroupNorm/InstanceNorm affine parameters
    cannot drift across iterations.  Fragile ViT structural parameters remain
    excluded by name as before.
    """
    module_map = dict(model.named_modules())
    norm_types = (
        nn.modules.batchnorm._BatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    fragile_keywords = ("pos_embed", "cls_token", "patch_embed")

    params = []
    group_ids = []
    module_to_gid = {}
    skipped_norm_tensors = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.rsplit(".", 1)
        module_name = parts[0] if len(parts) > 1 else name
        owner = module_map.get(module_name)
        if isinstance(owner, norm_types):
            skipped_norm_tensors += 1
            continue
        if any(keyword in name for keyword in fragile_keywords):
            continue
        if module_name not in module_to_gid:
            module_to_gid[module_name] = len(module_to_gid)
        params.append(param)
        group_ids.append(module_to_gid[module_name])
    return params, group_ids, len(module_to_gid), skipped_norm_tensors


def _project_group_l2(params, anchors, group_ids, num_groups, radius):
    """Project persistent parameter deltas onto per-module-group L2 balls.

    Returns the number of groups clipped on this update.  The projection is
    performed in-place under torch.no_grad() and allocates no persistent delta
    tensors.
    """
    if not params:
        return 0

    device = params[0].device
    group_sq = torch.zeros(num_groups, device=device, dtype=torch.float32)
    for param, anchor, gid in zip(params, anchors, group_ids):
        delta = param.detach() - anchor
        group_sq[gid] += delta.float().square().sum()

    group_norm = torch.sqrt(group_sq)
    clipped = group_norm > float(radius)
    clip_count = int(clipped.sum().item())
    if clip_count == 0:
        return 0

    scales = torch.ones_like(group_norm)
    scales[clipped] = float(radius) / (group_norm[clipped] + EPS)
    with torch.no_grad():
        for param, anchor, gid in zip(params, anchors, group_ids):
            if bool(clipped[gid].item()):
                scale = scales[gid].to(dtype=param.dtype)
                # Convert param to its delta in-place, scale, then re-anchor.
                param.sub_(anchor)
                param.mul_(scale)
                param.add_(anchor)
    return clip_count


def _restore_exact(params, anchors):
    """Restore selected source parameters exactly to the batch-entry state."""
    with torch.no_grad():
        for param, anchor in zip(params, anchors):
            param.copy_(anchor)
    return all(torch.equal(param.detach(), anchor) for param, anchor in zip(params, anchors))


def stateful_param_walk_attack(
    source_models,
    x,
    target_label,
    exp_settings,
    device,
    attack_type="RTMF",
    num_iter=300,
    max_epsilon=16,
    mu=1.0,
    step_size=0.002,
    radius=0.05,
    compute_stats=None,
    memory_checkpoint=None,
):
    """Run the frozen M=1 RTMF/FTM attack with a persistent parameter walk."""
    if attack_type != "RTMF":
        raise ValueError("P0-7A SPW is frozen to RTMF")
    if len(source_models) != 1 or not exp_settings.get("targeted", False):
        raise ValueError("P0-7A SPW requires one targeted source model")
    if num_iter < 2 or mu != 1.0:
        raise ValueError("P0-7A SPW requires >=2 nominal steps and momentum 1")
    if step_size < 0.0 or radius <= 0.0:
        raise ValueError("SPW step_size must be nonnegative and radius positive")
    if step_size > radius:
        raise ValueError("SPW step_size must not exceed the checkpoint radius")
    # step_size == 0 is permitted only as the P0-7 null-path audit control.
    # Positive SPW runs remain frozen to radius/(num_iter-1).
    expected_step = float(radius) / float(num_iter - 1)
    if (abs(float(step_size)) > 1e-12
            and abs(float(step_size) - expected_step) > 1e-12):
        raise ValueError(
            f"P0-7A-v2 requires step_size=0 for null audit or "
            f"step_size=radius/(num_iter-1)={expected_step:.12g}"
        )

    for source_model in source_models:
        source_model.eval()

    # Persistent state needs a stricter selector than historical temporary
    # D-LAP perturbations: normalization affine parameters are excluded by
    # actual module type, not by a fragile name substring.
    walk_params, group_ids, num_groups, skipped_norm_tensors = (
        _collect_spw_param_groups(source_models[0])
    )
    if not walk_params or num_groups < 1:
        raise ValueError("SPW found no eligible trainable source parameters")
    _count_compute(compute_stats, "spw_selected_groups", num_groups)
    _count_compute(
        compute_stats, "spw_skipped_normalization_tensors", skipped_norm_tensors
    )

    # One immutable checkpoint-centered anchor per selected parameter.
    anchors = [param.detach().clone() for param in walk_params]

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
    models = []

    try:
        # Match the historical FTM clean-feature cache exactly at theta_0.
        with torch.no_grad():
            models = [
                FeatureTuning(sm, x.size(-1), exp_settings, device)
                for sm in source_models
            ]
            for model in models:
                model.start_feature_record()
                model(x)
                _count_compute(compute_stats, "source_model_forward_executions")
                _count_compute(compute_stats, "clean_feature_cache_forwards")
                model.end_feature_record()

        if memory_checkpoint is not None:
            memory_checkpoint("after_feature_cache")

        # FTM consumes one nominal iteration for the clean cache, leaving 299
        # effective image/parameter-gradient steps for the frozen protocol.
        for step_index in range(1, num_iter):
            gradients_before_step = (
                compute_stats.get("image_gradient_evaluations", 0)
                if compute_stats is not None else 0
            )

            x_nes = x_adv.detach().requires_grad_(True)
            x_step = apply_input_transform(x_nes, attack_type, prob)
            total_loss = model_loss_sum(models, x_step, loss_fn, compute_stats)

            grad_targets = [x_nes]
            active_layers = []
            num_feature_targets = 0
            for model in models:
                tuning_params = []
                active_indices = []
                for layer_idx, was_triggered in model.mixing_triggered.items():
                    if was_triggered:
                        tuning_params.append(model.outputs_tuning[layer_idx])
                        active_indices.append(layer_idx)
                grad_targets.extend(tuning_params)
                active_layers.append((active_indices, len(tuning_params)))
                num_feature_targets += len(tuning_params)

            # Parameter gradients are additional outputs of the same backward,
            # not a second backward or a second stochastic surrogate sample.
            grad_targets.extend(walk_params)
            all_grads = torch.autograd.grad(
                total_loss,
                grad_targets,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            _count_compute(compute_stats, "backward_calls")
            _count_compute(compute_stats, "image_gradient_evaluations")
            _count_compute(compute_stats, "parameter_gradient_evaluations")

            grad_x = all_grads[0]

            # Preserve the ordinary FTM tuning-state update exactly once using
            # the feature gradients from this same sample.
            offset = 1
            for model_idx, (active_indices, count) in enumerate(active_layers):
                model = models[model_idx]
                model_grads = all_grads[offset:offset + count]
                for layer_idx, grad in zip(active_indices, model_grads):
                    if grad is None:
                        raise RuntimeError(
                            "SPW encountered an unused active FTM tuning tensor"
                        )
                    model.outputs_tuning[layer_idx] = (
                        model.outputs_tuning[layer_idx] - grad
                    ).detach().requires_grad_(True)
                offset += count

            theta_grads = all_grads[1 + num_feature_targets:]
            normalized_theta_grads = _normalized_gradient_per_group(
                theta_grads, group_ids, num_groups, device
            )

            del total_loss, x_step, grad_targets, all_grads, active_layers

            # Frozen FTM image path: TI -> L1 normalization -> MI momentum -> sign.
            processed = F.conv2d(
                grad_x,
                gaussian_kernel,
                bias=None,
                stride=1,
                padding=2,
                groups=3,
            )
            increment = processed / torch.sum(
                torch.abs(processed), dim=[1, 2, 3], keepdim=True
            )
            momentum = mu * momentum + increment
            x_adv = x_adv.detach() + alpha * momentum.sign()
            x_adv = torch.clamp(x_adv, x_min, x_max)

            _count_compute(compute_stats, "adversarial_image_updates")
            _count_compute(compute_stats, "effective_attack_steps")
            _count_compute(compute_stats, "m1_image_update_steps")
            _count_compute(compute_stats, "spw_steps")

            # Persistent target-adverse parameter move.  The update is made
            # only after grad_x_t has been consumed, so its first effect is on
            # the next source forward.  The terminal update is harmless and is
            # discarded by the exact batch-end restore.
            with torch.no_grad():
                for param, direction in zip(walk_params, normalized_theta_grads):
                    if direction is not None:
                        param.add_(direction, alpha=-float(step_size))
            _count_compute(compute_stats, "spw_parameter_updates")

            clipped_groups = _project_group_l2(
                walk_params, anchors, group_ids, num_groups, radius
            )
            if clipped_groups:
                _count_compute(
                    compute_stats, "spw_projection_clips", clipped_groups
                )

            if compute_stats is not None:
                compute_stats.setdefault("samples_per_step", []).append(
                    compute_stats.get("image_gradient_evaluations", 0)
                    - gradients_before_step
                )

            effective_step = step_index
            if memory_checkpoint is not None and effective_step in (
                1, 10, 50, num_iter - 1
            ):
                memory_checkpoint("after_attack_step", effective_step)

            del grad_x, processed, increment, theta_grads, normalized_theta_grads

    finally:
        # A batch must never leak its virtual surrogate state into the next
        # batch.  copy_ from immutable anchors gives an exact restoration.
        restored = _restore_exact(walk_params, anchors)
        _count_compute(compute_stats, "spw_restore_checks")
        if not restored:
            raise AssertionError("SPW failed to restore source parameters exactly")
        for model in models:
            model.remove_hooks()
        del anchors

    return x_adv.detach()
