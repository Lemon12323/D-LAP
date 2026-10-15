"""
LAP / D-LAP: loss-aware parameter perturbation for transferable targeted attacks.

Each attack iteration perturbs the surrogate weights inside a local parameter
neighborhood, estimates the input gradient under the perturbed model, then
updates the adversarial example with the same momentum-sign rule as FTM.

When LAP is enabled, the perturbation radius is adapted per layer group based
on gradient statistics. When D-LAP is enabled, the random perturbation
direction is biased toward a target-loss-aware EMA direction.
"""

import hashlib
import torch
import torch.nn.functional as F
import numpy as np

from attacks import FeatureTuning, LogitLoss, DI, RDI, gkern, _count_compute


def _is_sensitive_vit_param(name):
    """
    Skip normalization layers (BatchNorm/LayerNorm scales and shifts) and 
    fragile ViT parameters (pos_embed, cls_token, patch_embed) to ensure 
    that running statistics and basic structure remain perfectly stable 
    during parameter-space perturbation.
    """
    sensitive_keywords = ["pos_embed", "cls_token", "patch_embed", "norm"]
    return any(k in name for k in sensitive_keywords)


def _collect_param_groups(model, layer_selector=None):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if layer_selector == "high":
        named = named[len(named) // 2 :]
    elif layer_selector == "mlp":
        named = [(n, p) for n, p in named if "mlp" in n.lower() or "fc" in n.lower()]

    params = []
    group_ids = []
    module_to_gid = {}
    gid_counter = 0

    for name, param in named:
        if _is_sensitive_vit_param(name):
            continue
        parts = name.rsplit(".", 1)
        module_name = parts[0] if len(parts) > 1 else name
        if module_name not in module_to_gid:
            module_to_gid[module_name] = gid_counter
            gid_counter += 1
        params.append(param)
        group_ids.append(module_to_gid[module_name])

    return params, group_ids, gid_counter


def _sample_global_direction(params):
    """Sample a unit-norm random direction in the full weight space."""
    directions = [torch.randn_like(p) for p in params]
    flat = torch.cat([d.reshape(-1) for d in directions])
    norm = flat.norm(p=2) + 1e-12
    return [d / norm for d in directions]


def _sample_per_group_direction(params, group_ids, num_groups):
    """Sample unit-norm random directions independently for each layer group."""
    directions = [torch.randn_like(p) for p in params]
    group_sq = torch.zeros(num_groups, device=params[0].device)
    for d, gid in zip(directions, group_ids):
        group_sq[gid] += (d * d).sum()
    group_norm = torch.sqrt(group_sq) + 1e-12
    for i, (d, gid) in enumerate(zip(directions, group_ids)):
        directions[i] = d / group_norm[gid]
    return directions


def _perturb_weights(params, direction, rho):
    """Apply parameter perturbation with either scalar or per-parameter radii."""
    if isinstance(rho, (int, float)):
        for p, d in zip(params, direction):
            p.data.add_(d, alpha=rho)
    elif torch.is_tensor(rho):
        for p, d, r in zip(params, direction, rho):
            p.data.add_(d * r)
    else:
        for p, d, r in zip(params, direction, rho):
            p.data.add_(d, alpha=float(r))


def _restore_weights(params, direction, rho):
    """Undo the parameter perturbation after each Monte Carlo sample."""
    if isinstance(rho, (int, float)):
        for p, d in zip(params, direction):
            p.data.add_(d, alpha=-rho)
    elif torch.is_tensor(rho):
        for p, d, r in zip(params, direction, rho):
            p.data.add_(-d * r)
    else:
        for p, d, r in zip(params, direction, rho):
            p.data.add_(d, alpha=-float(r))


def _compute_group_norms(grads, group_ids, num_groups, device):
    """Aggregate parameter-gradient norms at the layer-group level on GPU."""
    sq = torch.zeros(num_groups, device=device)
    for g, gid in zip(grads, group_ids):
        if g is not None:
            sq[gid] += (g * g).sum()
    return torch.sqrt(sq)


def _normalize_per_group(tensors, group_ids, num_groups):
    group_sq = torch.zeros(num_groups, device=tensors[0].device)
    for t, gid in zip(tensors, group_ids):
        group_sq[gid] += (t * t).sum()
    group_norm = torch.sqrt(group_sq) + 1e-12
    return [t / group_norm[gid] for t, gid in zip(tensors, group_ids)]


def _normalized_gradient_per_group(grads, group_ids, num_groups, device):
    """Normalize each module group's current gradient before EMA insertion."""
    group_sq = torch.zeros(num_groups, device=device)
    for grad, gid in zip(grads, group_ids):
        if grad is not None:
            group_sq[gid] += (grad * grad).sum()
    group_norm = torch.sqrt(group_sq) + 1e-12
    return [None if grad is None else grad / group_norm[gid]
            for grad, gid in zip(grads, group_ids)]


def _normalize_global(tensors):
    flat_norm = torch.sqrt(sum((t * t).sum() for t in tensors)) + 1e-12
    return [t / flat_norm for t in tensors]


def _normalize_per_group_inplace(tensors, group_ids, num_groups):
    group_sq = torch.zeros(num_groups, device=tensors[0].device)
    for t, gid in zip(tensors, group_ids):
        group_sq[gid] += (t * t).sum()
    group_norm = torch.sqrt(group_sq) + 1e-12
    for t, gid in zip(tensors, group_ids):
        t.div_(group_norm[gid])


def _normalize_global_inplace(tensors):
    flat_norm = torch.sqrt(sum((t * t).sum() for t in tensors)) + 1e-12
    for t in tensors:
        t.div_(flat_norm)


def _make_depthwise_gaussian_kernel(kernel_size, device):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    return torch.from_numpy(
        np.expand_dims(np.stack([kernel, kernel, kernel]), 1)
    ).to(device)


def _clear_cached_attention_biases(model):
    for module in model.modules():
        if hasattr(module, "ab"):
            delattr(module, "ab")
        cache = getattr(module, "attention_bias_cache", None)
        if isinstance(cache, dict):
            cache.clear()


def dlap_ftm_attack(
    source_models,
    x,
    y,
    target_label=None,
    exp_settings=None,
    device="cuda:0",
    attack_type="RTMF",
    num_iter=300,
    max_epsilon=16,
    mu=1.0,
    lap_rho=0.05,
    lap_rho_min=None,
    lap_layer_selector=None,
    lap_num_samples=1,
    lap_num_samples_interval=1,
    lap_grad_smooth_kernel=0,
    use_lap_adaptive_radius=False,
    lap_beta=0.9,
    lap_warmup=10,
    lap_cv_temp=1.0,
    lap_mode="cv",
    dlap_alpha=1.0,
    dlap_disable_annealing=False,
    control_mode="legacy",
    repeatgrad_validate=False,
    dlap_guidance_grad_norm="raw",
    dlap_guidance_refresh=1,
    dlap_late_guidance_refresh=1,
    dlap_state_refresh_interval=1,
    dlap_param_grad_samples_per_refresh=2,
    sparse_state_validate=False,
    dlap_restore_validate=False,
    compute_stats=None,
    memory_checkpoint=None,
    diagnostic_observer=None,
):
    """Run FTM with LAP / D-LAP parameter-space smoothing."""

    prob = exp_settings["p"]
    lr = exp_settings["alpha"]
    assert exp_settings["targeted"], "Only targeted attack is supported"
    if lap_num_samples_interval < 1:
        raise ValueError("lap_num_samples_interval must be >= 1")
    if lap_grad_smooth_kernel < 0:
        raise ValueError("lap_grad_smooth_kernel must be >= 0")
    if lap_grad_smooth_kernel > 0 and (
        lap_grad_smooth_kernel < 3 or lap_grad_smooth_kernel % 2 == 0
    ):
        raise ValueError("lap_grad_smooth_kernel must be 0 or an odd integer >= 3")
    if lap_mode not in ("cv", "consistency"):
        raise ValueError("lap_mode must be one of: 'cv', 'consistency'")
    efficient_modes = ("late_m2_warmk4", "late_m1_warmk4", "late_m1_sparsek4")
    sampling_modes = ("sampling_full_m1", "sampling_interleave_k2", "sampling_interleave_k4")
    temporal_modes = ("temporal_antithetic_m1", "guidance_throughout_m1",
                      "temporal_antithetic_guidance_m1")
    if control_mode not in ("legacy", "repeatgrad_m2", "sparse_state_m2", "isotropic_sparse_m2") + efficient_modes + sampling_modes + temporal_modes:
        raise ValueError(f"Unsupported control_mode: {control_mode}")
    if dlap_guidance_grad_norm not in ("raw", "normalized"):
        raise ValueError("dlap_guidance_grad_norm must be raw or normalized")
    if dlap_guidance_refresh < 1 or dlap_late_guidance_refresh < 1:
        raise ValueError("Guidance refresh intervals must be positive")
    isotropic_mode = control_mode == "isotropic_sparse_m2"
    repeatgrad_mode = control_mode == "repeatgrad_m2"
    efficient_mode = control_mode in efficient_modes
    sampling_mode = control_mode in sampling_modes + temporal_modes
    antithetic_mode = control_mode in ("temporal_antithetic_m1", "temporal_antithetic_guidance_m1")
    guidance_throughout = control_mode in ("guidance_throughout_m1", "temporal_antithetic_guidance_m1")
    sparse_state_mode = control_mode in ("sparse_state_m2", "isotropic_sparse_m2") or sampling_mode
    if sparse_state_mode:
        expected_samples = 1 if control_mode == "sampling_full_m1" or control_mode in temporal_modes else 2
        if (lap_num_samples != expected_samples or lap_num_samples_interval != 1
                or dlap_guidance_grad_norm != "normalized"
                or (dlap_alpha != 1.0 if isotropic_mode else dlap_alpha >= 1.0)
                or dlap_disable_annealing or not use_lap_adaptive_radius
                or dlap_state_refresh_interval not in (1, 2, 4)
                or dlap_param_grad_samples_per_refresh not in (1, 2)):
            raise ValueError("Sparse-state mode requires its nominal M, normalized adaptive D-LAP, and valid K/S")
        if (sampling_mode or isotropic_mode) and (dlap_state_refresh_interval != 4
                              or dlap_param_grad_samples_per_refresh != 1):
            raise ValueError("Sampling modes preserve B5 parameter-state refresh K=4,S=1")
    elif (dlap_state_refresh_interval != 1 or dlap_param_grad_samples_per_refresh != 2
          or sparse_state_validate):
        raise ValueError("State refresh controls require sparse_state_m2")
    if repeatgrad_mode and lap_num_samples != 2:
        raise ValueError("repeatgrad_m2 requires lap_num_samples=2")
    if not efficient_mode and (dlap_guidance_refresh != 1 or dlap_late_guidance_refresh != 1):
        raise ValueError("Sparse guidance refresh requires an explicit efficient mode")
    if efficient_mode:
        expected_samples = 2 if control_mode == "late_m2_warmk4" else 1
        expected_late_refresh = 4 if control_mode == "late_m1_sparsek4" else 1
        if (lap_num_samples != expected_samples or dlap_guidance_refresh != 4
                or dlap_late_guidance_refresh != expected_late_refresh
                or dlap_guidance_grad_norm != "normalized" or dlap_alpha >= 1.0):
            raise ValueError("Efficient mode requires its fixed M, K=4 warmup, late refresh, normalized guidance, and directional alpha")

    for source_model in source_models:
        # Strictly set to eval() mode to freeze running statistics (e.g., BN/LN).
        # This is critical to prevent weight perturbations from corrupting running means/vars.
        source_model.eval()

    ti_kernel_size = 5
    gaussian_kernel = None
    if "T" in attack_type:
        gaussian_kernel = _make_depthwise_gaussian_kernel(ti_kernel_size, device)

    update_smooth_kernel = None
    if lap_grad_smooth_kernel > 0:
        update_smooth_kernel = _make_depthwise_gaussian_kernel(
            lap_grad_smooth_kernel, device
        )

    eps = max_epsilon / 255.0
    alpha = lr / 255.0
    x_min = torch.clamp(x - eps, 0.0, 1.0)
    x_max = torch.clamp(x + eps, 0.0, 1.0)
    x_adv = x.clone()
    g = 0
    loss_fn = LogitLoss(target_label, exp_settings["targeted"])

    consumed_iteration = 0
    if "F" in attack_type:
        with torch.no_grad():
            img_width = x.size()[-1]
            models = []
            for source_model in source_models:
                models.append(FeatureTuning(source_model, img_width, exp_settings, device))
            for model in models:
                model.start_feature_record()
                model(x)
                _count_compute(compute_stats, 'source_model_forward_executions')
                _count_compute(compute_stats, 'clean_feature_cache_forwards')
                model.end_feature_record()
            consumed_iteration = 1
            assert consumed_iteration < num_iter
    else:
        models = source_models

    if memory_checkpoint is not None:
        memory_checkpoint("after_feature_cache")

    all_model_params = []
    all_group_ids = []
    total_groups = 0
    for sm in source_models:
        params, gids, ng = _collect_param_groups(sm, lap_layer_selector)
        gids_offset = [gid + total_groups for gid in gids]
        all_model_params.extend(params)
        all_group_ids.extend(gids_offset)
        total_groups += ng

    all_group_ids_tensor = None
    if len(all_group_ids) > 0:
        all_group_ids_tensor = torch.tensor(all_group_ids, dtype=torch.long, device=device)

    use_adaptive_rho = (lap_rho_min is not None) and (lap_rho_min < lap_rho)
    total_steps = num_iter - consumed_iteration
    if use_adaptive_rho:
        log_ratio = np.log(lap_rho_min / lap_rho)

    lap_iter_count = 0
    if use_lap_adaptive_radius:
        ema_norm = torch.zeros(total_groups, device=device)
        ema_norm_sq = torch.zeros(total_groups, device=device)
        ema_consistency = torch.ones(total_groups, device=device)
        use_consistency_mode = (
            lap_mode == "consistency"
            and lap_num_samples > 1
            and len(all_model_params) > 0
            and total_groups > 0
        )
        if use_consistency_mode:
            grad_sum_params = [torch.zeros_like(p) for p in all_model_params]
            group_norm_sum = torch.zeros(total_groups, device=device)

    # The isotropic control retains existing EMA work and adaptive-radius gradients;
    # its perturbation directions never read the directional EMA.
    use_dlap = (dlap_alpha < 1.0 or isotropic_mode) and not repeatgrad_mode
    if use_dlap:
        ema_theta_grads = [torch.zeros_like(p) for p in all_model_params]

    antithetic_cache = None

    repeatgrad_reference = None
    repeatgrad_versions = None
    guidance_refreshes = 0
    if repeatgrad_mode and repeatgrad_validate:
        repeatgrad_reference = [p.detach().clone() for p in all_model_params]
        repeatgrad_versions = [p._version for p in all_model_params]

    for t in range(consumed_iteration, num_iter):
        local_step = t - consumed_iteration
        if isotropic_mode and compute_stats is not None:
            compute_stats.setdefault("isotropic_alpha_by_step", []).append(1.0)
        progress = local_step / max(total_steps - 1, 1)
        late_phase = progress > 0.6
        perturb_this_step = not repeatgrad_mode and (not efficient_mode or late_phase)
        if efficient_mode:
            _count_compute(compute_stats, 'late_phase_image_updates' if late_phase
                           else 'early_phase_image_updates')
        if use_adaptive_rho:
            rho_t = lap_rho * np.exp(log_ratio * progress)
        else:
            rho_t = lap_rho
        if compute_stats is not None and control_mode in temporal_modes:
            compute_stats.setdefault('dlap_rho_by_step', []).append(float(rho_t))

        if use_lap_adaptive_radius and lap_iter_count >= lap_warmup:
            var = ema_norm_sq - ema_norm**2
            std = torch.sqrt(torch.clamp(var, min=0)) + 1e-12
            mean = ema_norm + 1e-12
            cv = std / mean
            if lap_mode == "consistency":
                cv_scaled = cv if lap_cv_temp == 1.0 else cv**lap_cv_temp
                score = ema_consistency / (cv_scaled + 1e-12)
                score_ratio = score / (score.mean() + 1e-12)
            else:
                cv_ratio = cv / (cv.mean() + 1e-12)
                if lap_cv_temp != 1.0:
                    cv_ratio = cv_ratio**lap_cv_temp
                    cv_ratio = cv_ratio / (cv_ratio.mean() + 1e-12)
                score_ratio = cv_ratio
            score_ratio = torch.clamp(score_ratio, max=3.0)
            rho_per_param = (rho_t * score_ratio)[all_group_ids_tensor]
        else:
            rho_per_param = rho_t

        if diagnostic_observer is not None:
            diagnostic_observer.begin_step(
                local_step, x_adv, g, rho_t, rho_per_param,
                ema_norm if use_lap_adaptive_radius else None,
                ema_norm_sq if use_lap_adaptive_radius else None,
                lap_iter_count, gaussian_kernel)

        num_samples_t = 1 if efficient_mode and not late_phase else lap_num_samples
        if lap_num_samples > 1 and lap_num_samples_interval > 1:
            if (local_step % lap_num_samples_interval) != 0:
                num_samples_t = 1
        if control_mode == "sampling_interleave_k2":
            num_samples_t = 2 if local_step % 2 == 0 else 1
        elif control_mode == "sampling_interleave_k4":
            num_samples_t = 2 if local_step % 4 == 0 else 1
        _count_compute(compute_stats, 'm2_image_update_steps' if num_samples_t == 2
                       else 'm1_image_update_steps')

        parameter_state_refresh = sparse_state_mode and local_step % dlap_state_refresh_interval == 0
        if sparse_state_mode:
            _count_compute(compute_stats, 'parameter_state_refresh_steps' if parameter_state_refresh
                           else 'parameter_state_nonrefresh_steps')
            if parameter_state_refresh and compute_stats is not None:
                compute_stats.setdefault('parameter_state_refresh_step_indices', []).append(local_step)
        if sparse_state_validate and not parameter_state_refresh:
            frozen_state_tensors = ema_theta_grads + [ema_norm, ema_norm_sq, ema_consistency]
            if use_consistency_mode:
                frozen_state_tensors += grad_sum_params + [group_norm_sum]
            frozen_state_versions = [tensor._version for tensor in frozen_state_tensors]

        grad_x_acc = None
        if use_lap_adaptive_radius and perturb_this_step:
            lap_norm_acc = torch.zeros(total_groups, device=device)
            adaptive_grad_samples = 0
            use_consistency_t = use_consistency_mode and num_samples_t > 1
            if use_consistency_t and (not sparse_state_mode or parameter_state_refresh):
                group_norm_sum.zero_()
                for gsum in grad_sum_params:
                    gsum.zero_()

        for k in range(num_samples_t):
            is_last = k == num_samples_t - 1
            if perturb_this_step or repeatgrad_mode:
                if use_lap_adaptive_radius:
                    direction_k = _sample_per_group_direction(
                        all_model_params, all_group_ids, total_groups
                    )
                else:
                    direction_k = _sample_global_direction(all_model_params)
                _count_compute(compute_stats, 'random_direction_draw_calls')
                _count_compute(compute_stats, 'random_direction_tensor_draw_calls', len(all_model_params))
                if antithetic_mode:
                    if local_step % 2 == 0:
                        if antithetic_cache is not None:
                            raise AssertionError("Antithetic cache crossed a pair boundary")
                        _count_compute(compute_stats, 'antithetic_even_fresh_directions')
                        if local_step + 1 < total_steps:
                            antithetic_cache = [d.detach().clone() for d in direction_k]
                        else:
                            _count_compute(compute_stats, 'antithetic_unpaired_final_steps')
                    else:
                        if antithetic_cache is None:
                            raise AssertionError("Missing antithetic exploration direction")
                        # Keep the vanilla draw above for its RNG effect only. The
                        # previous even step's exploration is negated before any
                        # guidance mixing or final direction normalization.
                        del direction_k
                        _count_compute(compute_stats, 'antithetic_rng_discarded_directions')
                        direction_k = [-d for d in antithetic_cache]
                        if sparse_state_validate:
                            dot = sum((a * b).sum() for a, b in zip(antithetic_cache, direction_k))
                            norm_a = sum((a * a).sum() for a in antithetic_cache)
                            norm_b = sum((b * b).sum() for b in direction_k)
                            cosine = (dot / torch.sqrt(norm_a * norm_b)).item()
                            if abs(cosine + 1.0) > 1e-5:
                                raise AssertionError("Antithetic exploration cosine differs from -1")
                            if compute_stats is not None:
                                compute_stats['antithetic_max_cosine_error'] = max(
                                    compute_stats.get('antithetic_max_cosine_error', 0.0), abs(cosine + 1.0))
                        del antithetic_cache
                        antithetic_cache = None
                        _count_compute(compute_stats, 'antithetic_odd_cached_directions')
                        _count_compute(compute_stats, 'antithetic_completed_pairs')

            need_param_grads = (use_lap_adaptive_radius or use_dlap or repeatgrad_mode)
            if efficient_mode:
                refresh_interval = (dlap_late_guidance_refresh if late_phase
                                    else dlap_guidance_refresh)
                need_param_grads = local_step % refresh_interval == 0
            if sparse_state_mode:
                need_param_grads = parameter_state_refresh and k < dlap_param_grad_samples_per_refresh
                if need_param_grads:
                    _count_compute(compute_stats, 'parameter_state_sample1_grads' if k == 0
                                   else 'parameter_state_sample2_grads')

            # Consume the same random-direction draws as LAP/D-LAP so that
            # subsequent RTMF transformations follow the matched RNG stream.
            # RepeatGrad never applies this direction to model parameters.

            if use_dlap and perturb_this_step and (lap_iter_count > 0 or guidance_throughout or
                                                   (efficient_mode and guidance_refreshes > 0)):
                if guidance_throughout or dlap_disable_annealing:
                    current_alpha = dlap_alpha
                else:
                    progress = (t - consumed_iteration) / max(total_steps - 1, 1)
                    explore_ratio = 0.6
                    if progress <= explore_ratio:
                        current_alpha = 1.0
                    else:
                        decay_progress = (progress - explore_ratio) / (1.0 - explore_ratio)
                        current_alpha = 1.0 - decay_progress * (1.0 - dlap_alpha)

                if compute_stats is not None and control_mode in temporal_modes:
                    compute_stats.setdefault('dlap_alpha_by_step', []).append(current_alpha)

                if not isotropic_mode:
                    if use_lap_adaptive_radius:
                        guided_dir = _normalize_per_group(
                            ema_theta_grads, all_group_ids, total_groups
                        )
                    else:
                        guided_dir = _normalize_global(ema_theta_grads)

                    for rnd, gd in zip(direction_k, guided_dir):
                        rnd.mul_(current_alpha).sub_(gd, alpha=1.0 - current_alpha)

                    del guided_dir
                if use_lap_adaptive_radius:
                    _normalize_per_group_inplace(direction_k, all_group_ids, total_groups)
                else:
                    _normalize_global_inplace(direction_k)
            elif use_dlap and perturb_this_step and compute_stats is not None and control_mode in temporal_modes:
                # The original first step uses the random exploration alone.
                compute_stats.setdefault('dlap_alpha_by_step', []).append(1.0)

            if perturb_this_step:
                if sparse_state_validate or dlap_restore_validate:
                    before_perturb = [p.detach().clone() for p in all_model_params]
                _perturb_weights(all_model_params, direction_k, rho_per_param)
                _count_compute(compute_stats, 'sampled_parameter_neighborhoods')
                _count_compute(compute_stats, 'parameter_perturbations_applied')
                if efficient_mode:
                    _count_compute(compute_stats, 'late_phase_parameter_perturbations' if late_phase
                                   else 'early_phase_parameter_perturbations')

            x_nes = x_adv.detach()
            x_nes.requires_grad_(True)

            if "D" in attack_type:
                x_in = DI(x_nes, prob)
            elif "R" in attack_type:
                x_in = RDI(x_nes)
                _count_compute(compute_stats, 'rtmf_transform_calls')
                _count_compute(compute_stats, 'rtmf_transform_random_scalar_draws', 3)
            else:
                x_in = x_nes

            total_loss = 0
            _count_compute(compute_stats, 'logical_attack_forward_calls')
            for model in models:
                _clear_cached_attention_biases(model)
                model_logits = model(x_in)
                total_loss += loss_fn(model_logits)
                if diagnostic_observer is not None and k == 0:
                    diagnostic_observer.record_sample1_logits(model_logits.detach())
                _count_compute(compute_stats, 'source_model_forward_executions')
                if "F" in attack_type:
                    decisions = len(model.mixing_triggered)
                    triggered = sum(bool(value) for value in model.mixing_triggered.values())
                    _count_compute(compute_stats, 'ftm_mixing_decision_draw_calls', decisions)
                    _count_compute(compute_stats, 'ftm_shuffle_draw_calls',
                                   triggered if exp_settings['shuffle_image_feature'] == 'SelfShuffle' else 0)
                    _count_compute(compute_stats, 'ftm_coefficient_draw_calls', triggered)
                del model_logits

            if "F" in attack_type:
                grad_targets = [x_nes]
                active_layers_info = []
                num_feat = 0
                need_feature_grads = is_last or use_dlap or repeatgrad_mode
                if need_feature_grads:
                    for model in models:
                        tuning_ps = []
                        active_idxs = []
                        for li, triggered in model.mixing_triggered.items():
                            if triggered:
                                tuning_ps.append(model.outputs_tuning[li])
                                active_idxs.append(li)
                        grad_targets.extend(tuning_ps)
                        active_layers_info.append((active_idxs, len(tuning_ps)))
                        num_feat += len(tuning_ps)

                if need_param_grads:
                    grad_targets.extend(all_model_params)

                all_grads = torch.autograd.grad(
                    total_loss,
                    grad_targets,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=need_param_grads,
                )
                _count_compute(compute_stats, 'backward_calls')
                _count_compute(compute_stats, 'image_gradient_evaluations')
                if need_param_grads:
                    _count_compute(compute_stats, 'parameter_gradient_evaluations')
                grad_x_k = all_grads[0]

                if need_feature_grads and is_last:
                    offset = 1
                    for model_idx, (active_idxs, n_p) in enumerate(active_layers_info):
                        mdl = models[model_idx]
                        for i, li in enumerate(active_idxs):
                            fg = all_grads[offset + i]
                            mdl.outputs_tuning[li] = (
                                mdl.outputs_tuning[li] - fg
                            ).detach().requires_grad_(True)
                        offset += n_p

                if need_param_grads:
                    theta_grads = all_grads[1 + num_feat :]

                    if use_lap_adaptive_radius and perturb_this_step:
                        norms = _compute_group_norms(
                            theta_grads, all_group_ids, total_groups, device
                        )
                        lap_norm_acc.add_(norms)
                        adaptive_grad_samples += 1
                        if use_consistency_t:
                            group_norm_sum.add_(norms)
                            for gsum, tg in zip(grad_sum_params, theta_grads):
                                if tg is not None:
                                    gsum.add_(tg)

            else:
                grad_targets = [x_nes]
                if need_param_grads:
                    grad_targets.extend(all_model_params)

                all_grads = torch.autograd.grad(
                    total_loss,
                    grad_targets,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=need_param_grads,
                )
                _count_compute(compute_stats, 'backward_calls')
                _count_compute(compute_stats, 'image_gradient_evaluations')
                if need_param_grads:
                    _count_compute(compute_stats, 'parameter_gradient_evaluations')
                grad_x_k = all_grads[0]

                if need_param_grads:
                    theta_grads = all_grads[1:]

                    if use_lap_adaptive_radius and perturb_this_step:
                        norms = _compute_group_norms(
                            theta_grads, all_group_ids, total_groups, device
                        )
                        lap_norm_acc.add_(norms)
                        adaptive_grad_samples += 1
                        if use_consistency_t:
                            group_norm_sum.add_(norms)
                            for gsum, tg in zip(grad_sum_params, theta_grads):
                                if tg is not None:
                                    gsum.add_(tg)

            if use_dlap and need_param_grads:
                normalized_grads = (_normalized_gradient_per_group(
                    theta_grads, all_group_ids, total_groups, device
                ) if dlap_guidance_grad_norm == "normalized" else theta_grads)
                for ema_g, tg in zip(ema_theta_grads, normalized_grads):
                    if tg is not None:
                        ema_g.mul_(lap_beta).add_(tg, alpha=1 - lap_beta)
                del normalized_grads
                if isotropic_mode:
                    _count_compute(compute_stats, "unused_directional_memory_updates")
                    _count_compute(compute_stats, "unused_directional_memory_gradient_normalizations")
                    _count_compute(compute_stats, "unused_directional_memory_tensor_updates",
                                   sum(tg is not None for tg in theta_grads))
                guidance_refreshes += 1
                _count_compute(compute_stats, 'guidance_refresh_count')
                _count_compute(compute_stats, 'late_guidance_refresh_count' if late_phase
                               else 'early_guidance_refresh_count')

            if diagnostic_observer is not None:
                if k == 0:
                    diagnostic_observer.first_sample(grad_x_k.detach())
                elif k == 1:
                    diagnostic_observer.second_sample(
                        grad_x_acc.detach(), grad_x_k.detach())

            if perturb_this_step:
                _restore_weights(all_model_params, direction_k, rho_per_param)
                if sparse_state_validate or dlap_restore_validate:
                    if not all(torch.allclose(p, original, rtol=1e-6, atol=1e-6)
                               for p, original in zip(all_model_params, before_perturb)):
                        raise AssertionError("Source parameters were not restored after a neighborhood sample")
                    _count_compute(compute_stats, 'parameter_restore_checks')
                    del before_perturb
            elif repeatgrad_validate:
                if any(p._version != version for p, version in zip(all_model_params, repeatgrad_versions)):
                    raise AssertionError("repeatgrad_m2 changed a surrogate parameter version")

            del all_grads, grad_targets, x_in, total_loss
            if need_param_grads:
                del theta_grads
            if perturb_this_step or repeatgrad_mode:
                del direction_k

            grad_x_acc = grad_x_k if grad_x_acc is None else grad_x_acc + grad_x_k

        grad_x = grad_x_acc / num_samples_t

        if use_lap_adaptive_radius and perturb_this_step and adaptive_grad_samples:
            if sparse_state_mode:
                _count_compute(compute_stats, 'parameter_state_adaptive_updates')
            norms_iter = lap_norm_acc / adaptive_grad_samples
            ema_norm.mul_(lap_beta).add_(norms_iter, alpha=1 - lap_beta)
            ema_norm_sq.mul_(lap_beta).add_(norms_iter * norms_iter, alpha=1 - lap_beta)
            if lap_mode == "consistency" and use_consistency_t:
                group_sum_sq = torch.zeros(total_groups, device=device)
                for gsum, gid in zip(grad_sum_params, all_group_ids):
                    group_sum_sq[gid] += (gsum * gsum).sum()
                consistency_samples = (adaptive_grad_samples if sparse_state_mode else num_samples_t)
                mean_grad_norm = torch.sqrt(group_sum_sq) / max(consistency_samples, 1)
                mean_norm = group_norm_sum / max(consistency_samples, 1) + 1e-12
                consistency_iter = torch.clamp(mean_grad_norm / mean_norm, min=0.0)
                ema_consistency.mul_(lap_beta).add_(consistency_iter, alpha=1 - lap_beta)
        if sparse_state_mode and not parameter_state_refresh and adaptive_grad_samples != 0:
            raise AssertionError("Non-refresh step updated parameter-gradient-dependent state")
        if sparse_state_validate and not parameter_state_refresh:
            if any(tensor._version != version for tensor, version in
                   zip(frozen_state_tensors, frozen_state_versions)):
                raise AssertionError("Non-refresh step wrote parameter-gradient-dependent state")
            _count_compute(compute_stats, 'parameter_state_freeze_checks')
        if (use_lap_adaptive_radius and perturb_this_step) or (use_dlap and not efficient_mode):
            lap_iter_count += 1

        if "T" in attack_type:
            grad_x = F.conv2d(
                grad_x,
                gaussian_kernel,
                bias=None,
                stride=1,
                padding=((ti_kernel_size - 1) // 2, (ti_kernel_size - 1) // 2),
                groups=3,
            )

        if "M" in attack_type:
            g = mu * g + grad_x / torch.sum(
                torch.abs(grad_x), dim=[1, 2, 3], keepdim=True
            )
        else:
            g = grad_x

        update_direction = g
        if update_smooth_kernel is not None:
            update_direction = F.conv2d(
                update_direction,
                update_smooth_kernel,
                bias=None,
                stride=1,
                padding=(
                    (lap_grad_smooth_kernel - 1) // 2,
                    (lap_grad_smooth_kernel - 1) // 2,
                ),
                groups=3,
            )

        pert = alpha * update_direction.sign()
        x_adv = x_adv.detach() + pert
        x_adv = torch.clamp(x_adv, x_min, x_max)
        _count_compute(compute_stats, 'adversarial_image_updates')
        _count_compute(compute_stats, 'effective_attack_steps')
        if compute_stats is not None:
            compute_stats.setdefault('samples_per_step', []).append(num_samples_t)
        effective_step = local_step + 1
        if memory_checkpoint is not None and effective_step in (1, 10, 50, total_steps):
            memory_checkpoint("after_attack_step", effective_step)
        if diagnostic_observer is not None and effective_step in (60, 120, 178, 250, 298):
            diagnostic_observer.checkpoint(
                effective_step, x_adv, models, all_model_params,
                all_group_ids, total_groups,
                ema_theta_grads if use_dlap else None,
                ema_norm if use_lap_adaptive_radius else None,
                ema_norm_sq if use_lap_adaptive_radius else None,
                lap_iter_count, lap_rho, lap_rho_min, lap_cv_temp,
                lap_mode, lap_warmup)

    if "F" in attack_type:
        for model in models:
            model.remove_hooks()
    if antithetic_cache is not None:
        raise AssertionError("Antithetic exploration cache survived the attack")
    if repeatgrad_validate:
        if not all(torch.equal(p.detach(), ref) for p, ref in zip(all_model_params, repeatgrad_reference)):
            raise AssertionError("repeatgrad_m2 changed surrogate parameter values")
        if compute_stats is not None:
            compute_stats["repeatgrad_parameter_integrity_validated"] = True
    if repeatgrad_mode and compute_stats is not None:
        compute_stats["parameter_perturbation_linf_max"] = 0.0
    if compute_stats is not None and (sparse_state_mode or dlap_guidance_grad_norm == "normalized"):
        compute_stats["rng_state_end_sha256"] = {
            "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "torch_cuda": hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest(),
            "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        }
    return x_adv.detach()

