import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from typing import Callable
from torch import nn, Tensor


def _count_compute(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount

def _legacy_ftm_attack(source_models, x, y, target_label=None, exp_settings=None, device='cuda:0', attack_type='RTMF', num_iter=300, max_epsilon=16, mu=1.0):
    prob = exp_settings['p']
    lr = exp_settings['alpha']
    assert exp_settings['targeted'] == True, "Not supported for non-targeted attack"

    for source_model in source_models:
        source_model.eval()

    ti_kernel_size = 5
    if 'T' in attack_type:
        kernel = gkern(ti_kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)

    eps = max_epsilon / 255.0
    alpha = lr / 255.0

    x_min = torch.clamp(x - eps, 0.0, 1.0)
    x_max = torch.clamp(x + eps, 0.0, 1.0)

    x_adv = x.clone()
    g = 0
    loss_fn = LogitLoss(target_label, exp_settings['targeted'])

    consumed_iteration = 0
    if 'F' in attack_type:  
        with torch.no_grad():
            img_width = x.size()[-1]
            x_f = x
            models = []
            for source_model in source_models:
                models.append(FeatureTuning(source_model, img_width, exp_settings, device))
            for model in models:
                model.start_feature_record()  
                model(x_f)  
                model.end_feature_record()  
            consumed_iteration = 1  
            assert consumed_iteration < num_iter, "consumed_iteration should be less than num_iter"
    elif 'C' in attack_type:
        with torch.no_grad():
            img_width = x.size()[-1]
            models = []
            for source_model in source_models:
                models.append(CleanFeatureMixup(source_model, img_width, exp_settings, device))
            for model in models:
                model.start_feature_record()
                model(x)
                model.end_feature_record()
            consumed_iteration = 1
            assert consumed_iteration < num_iter, "consumed_iteration should be less than num_iter"
    else:
        models = source_models

    for t in range(consumed_iteration, num_iter):
        x_nes = x_adv.detach()
        x_nes.requires_grad = True

        x_step = x_nes
        if 'D' in attack_type:
            x_step = DI(x_step, prob)
        elif 'R' in attack_type:
            x_step = RDI(x_step)
        elif 'O' in attack_type:
            x_step = ODI(x_step)

        if 'S' in attack_type and 'F' not in attack_type:
            scale_start = exp_settings.get('su_scale_start', 0.5)
            scale_interval = exp_settings.get('su_scale_interval', 0.2)
            x_step = SU_augment_single(x_step, scale_start, scale_interval)

        if 'A' in attack_type and 'F' not in attack_type:
            admix_alpha = exp_settings.get('admix_alpha', 0.2)
            j = torch.randint(0, 3, (1,)).item()
            scale = 1.0 / (2 ** j)
            batch_perm = torch.randperm(x.size(0), device=device)
            x_step = x_step + admix_alpha * x[batch_perm] * scale

        total_loss = 0
        for model in models:
            if 'C' in attack_type and 'F' not in attack_type:
                cfm_module.model = model
                out = cfm_module(x_step)
            else:
                out = model(x_step)
            total_loss += loss_fn(out)

        if 'F' in attack_type:
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

            all_grads = torch.autograd.grad(total_loss, all_params, retain_graph=False)
            grad_x = all_grads[0]

            current_idx = 1
            for model_idx, (active_indices, num_params) in enumerate(all_active_layers):
                model = models[model_idx]
                model_grads = all_grads[current_idx:current_idx + num_params]
                for layer_idx, grad in zip(active_indices, model_grads):
                    model.outputs_tuning[layer_idx] = (model.outputs_tuning[layer_idx] - grad).detach().requires_grad_(True)
                current_idx += num_params
        else:
            grad_x = torch.autograd.grad(total_loss, x_nes, retain_graph=False)[0]

        if 'T' in attack_type:
            grad_x = F.conv2d(grad_x, gaussian_kernel, bias=None, stride=1,
                              padding=((ti_kernel_size - 1) // 2, (ti_kernel_size - 1) // 2), groups=3)

        if 'M' in attack_type:
            g = mu * g + grad_x / torch.sum(torch.abs(grad_x), dim=[1, 2, 3], keepdim=True)
        else:
            g = grad_x

        pert = alpha * g.sign()
        x_adv = x_adv.detach() + pert
        x_adv = torch.clamp(x_adv, x_min, x_max)

    if 'F' in attack_type:
        for model in models: model.remove_hooks()
    if cfm_module is not None: cfm_module.remove_hooks()

    return x_adv.detach()


def ftm_attack(source_models, x, y, target_label=None, exp_settings=None, device='cuda:0', attack_type='RTMF', num_iter=300, max_epsilon=16, mu=1.0, compute_stats=None, memory_checkpoint=None):
    prob = exp_settings['p']
    lr = exp_settings['alpha']
    assert exp_settings['targeted'] == True, "Not supported for non-targeted attack"

    for source_model in source_models:
        source_model.eval()

    ti_kernel_size = 5
    if 'T' in attack_type:
        kernel = gkern(ti_kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)

    eps = max_epsilon / 255.0
    alpha = lr / 255.0
    x_min = torch.clamp(x - eps, 0.0, 1.0)
    x_max = torch.clamp(x + eps, 0.0, 1.0)

    x_adv = x.clone()
    g = 0
    loss_fn = LogitLoss(target_label, exp_settings['targeted'])

    consumed_iteration = 0
    if 'F' in attack_type:
        with torch.no_grad():
            img_width = x.size()[-1]
            models = [FeatureTuning(m, img_width, exp_settings, device) for m in source_models]
            for model in models:
                model.start_feature_record()
                model(x)
                _count_compute(compute_stats, 'source_model_forward_executions')
                _count_compute(compute_stats, 'clean_feature_cache_forwards')
                model.end_feature_record()
            consumed_iteration = 1
            assert consumed_iteration < num_iter, "consumed_iteration should be less than num_iter"
    elif 'C' in attack_type:
        with torch.no_grad():
            img_width = x.size()[-1]
            models = [CleanFeatureMixup(m, img_width, exp_settings, device) for m in source_models]
            for model in models:
                model.start_feature_record()
                model(x)
                _count_compute(compute_stats, 'source_model_forward_executions')
                _count_compute(compute_stats, 'clean_feature_cache_forwards')
                model.end_feature_record()
            consumed_iteration = 1
            assert consumed_iteration < num_iter, "consumed_iteration should be less than num_iter"
    else:
        models = source_models

    if memory_checkpoint is not None:
        memory_checkpoint('after_feature_cache')

    try:
        for step_index in range(consumed_iteration, num_iter):
            gradients_before_step = compute_stats.get('image_gradient_evaluations', 0) if compute_stats is not None else 0
            x_nes = x_adv.detach()
            x_nes.requires_grad = True

            if 'E' in attack_type and 'F' not in attack_type:
                grad_x = calculate_everywhere_grad(models, x_nes, attack_type, prob, loss_fn, exp_settings, compute_stats)
            elif 'A' in attack_type and 'F' not in attack_type:
                grad_x = calculate_admix_grad(models, x_nes, x, attack_type, prob, loss_fn, exp_settings, compute_stats)
            elif 'S' in attack_type and 'F' not in attack_type:
                grad_x = calculate_si_grad(models, x_nes, attack_type, prob, loss_fn, exp_settings, compute_stats)
            else:
                x_step = apply_input_transform(x_nes, attack_type, prob)
                total_loss = model_loss_sum(models, x_step, loss_fn, compute_stats)

                if 'F' in attack_type:
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

                    all_grads = torch.autograd.grad(total_loss, all_params, retain_graph=False, create_graph=False)
                    _count_compute(compute_stats, 'backward_calls')
                    _count_compute(compute_stats, 'image_gradient_evaluations')
                    grad_x = all_grads[0]

                    current_idx = 1
                    for model_idx, (active_indices, num_params) in enumerate(all_active_layers):
                        model = models[model_idx]
                        model_grads = all_grads[current_idx:current_idx + num_params]
                        for layer_idx, grad in zip(active_indices, model_grads):
                            model.outputs_tuning[layer_idx] = (model.outputs_tuning[layer_idx] - grad).detach().requires_grad_(True)
                        current_idx += num_params
                else:
                    grad_x = torch.autograd.grad(total_loss, x_nes, retain_graph=False, create_graph=False)[0]
                    _count_compute(compute_stats, 'backward_calls')
                    _count_compute(compute_stats, 'image_gradient_evaluations')
                del total_loss, x_step
                if 'F' in attack_type:
                    del all_grads, all_params, all_active_layers

            if 'T' in attack_type:
                grad_x = F.conv2d(grad_x, gaussian_kernel, bias=None, stride=1,
                                  padding=((ti_kernel_size - 1) // 2, (ti_kernel_size - 1) // 2), groups=3)

            if 'M' in attack_type:
                g = mu * g + grad_x / torch.sum(torch.abs(grad_x), dim=[1, 2, 3], keepdim=True)
            else:
                g = grad_x

            x_adv = x_adv.detach() + alpha * g.sign()
            x_adv = torch.clamp(x_adv, x_min, x_max)
            _count_compute(compute_stats, 'adversarial_image_updates')
            _count_compute(compute_stats, 'effective_attack_steps')
            if compute_stats is not None:
                compute_stats.setdefault('samples_per_step', []).append(
                    compute_stats.get('image_gradient_evaluations', 0) - gradients_before_step
                )
            effective_step = step_index - consumed_iteration + 1
            if memory_checkpoint is not None and effective_step in (1, 10, 50, num_iter - consumed_iteration):
                memory_checkpoint('after_attack_step', effective_step)
    finally:
        if 'F' in attack_type or 'C' in attack_type:
            for model in models:
                model.remove_hooks()

    return x_adv.detach()


class FeatureTuning(nn.Module):
    def __init__(self, model: nn.Module, input_size, exp_settings, device):
        super().__init__()
        self.exp_settings = exp_settings
        self.device = device
        self.mixup_layer = exp_settings['mixup_layer']
        self.prob = exp_settings['mix_prob']
        self.channelwise = exp_settings['channelwise']
        self.model = model
        self.input_size = input_size
        self.record = False
        self.cache_clean_features_cpu = bool(exp_settings.get('ftm_cache_clean_features_cpu', False))
        self.outputs = {}
        self.outputs_tuning = {}  
        self.mixing_triggered = {}
        self.forward_hooks = []
        # P0-8 diagnostics are opt-in so the ordinary FTM path remains
        # numerically and operationally unchanged.
        self.csr_track_feature_norms = bool(exp_settings.get('csr_track_feature_norms', False))
        self.current_feature_norms = {}

        def get_children(model: torch.nn.Module):
            children = list(model.children())
            flattened_children = []
            if children == []:
                if self.mixup_layer == 'conv_linear_no_last' or self.mixup_layer == 'conv_linear_include_last':
                    if type(model) == torch.nn.Conv2d or type(model) == torch.nn.Linear: return model
                    else: return []
                elif self.mixup_layer == 'bn' or self.mixup_layer == 'relu':
                    if type(model) == torch.nn.BatchNorm2d: return model
                    else: return []
                else:
                    if type(model) == torch.nn.Conv2d: return model
                    else: return []
            else:
                for child in children:
                    try: flattened_children.extend(get_children(child))
                    except TypeError: flattened_children.append(get_children(child))
            return flattened_children

        mod_list = get_children(model)
        # Expose the exact module attached to each FTM layer index. This is
        # read-only metadata used by P0-8 to pair an active tuning tensor with
        # the parameters of the same Conv/Linear module.
        self.layer_modules = list(mod_list)
        self.layer_num = len(mod_list)
        for i, m in enumerate(mod_list):
            self.forward_hooks.append(m.register_forward_hook(self.save_outputs_hook(i)))

    def save_outputs_hook(self, layer_idx) -> Callable:
        exp_settings = self.exp_settings
        mix_upper_bound_feature = exp_settings['mix_upper_bound_feature']
        mix_lower_bound_feature = exp_settings['mix_lower_bound_feature']
        shuffle_image_feature = exp_settings['shuffle_image_feature']
        blending_mode_feature = exp_settings['blending_mode_feature']
        mixed_image_type_feature = exp_settings['mixed_image_type_feature']
        divisor = exp_settings['divisor']

        def hook_fn(module, input, output):
            if type(module) == torch.nn.Linear or output.size()[-1] <= self.input_size // divisor:
                if self.mixup_layer == 'conv_linear_no_last' and (layer_idx + 1) == self.layer_num and type(module) == torch.nn.Linear:
                    pass  
                else:
                    if layer_idx in self.outputs and self.record == False:  
                        c = torch.rand(1).item()
                        self.mixing_triggered[layer_idx] = (c <= self.prob)
                        if self.mixing_triggered[layer_idx]:
                            if self.csr_track_feature_norms:
                                # Raw current feature norm, captured before FTM
                                # tuning/mixup modifies this layer's output.
                                self.current_feature_norms[layer_idx] = torch.linalg.vector_norm(output.detach())
                            if mixed_image_type_feature == 'A': prev_feature = output.clone().detach()
                            else:
                                prev_feature = self.outputs[layer_idx]
                                if prev_feature.device != output.device:
                                    prev_feature = prev_feature.to(output.device, non_blocking=True)
                            if shuffle_image_feature == 'SelfShuffle':  
                                idx = torch.randperm(output.shape[0])
                                prev_feature_shuffle = prev_feature[idx].view(prev_feature.size())
                            elif shuffle_image_feature == 'None':
                                prev_feature_shuffle = prev_feature
                            mix_ratio = mix_upper_bound_feature - mix_lower_bound_feature
                            if self.channelwise == True:
                                if output.dim() == 4: a = (torch.rand(prev_feature.shape[0], prev_feature.shape[1]) * mix_ratio + mix_lower_bound_feature).view(prev_feature.shape[0], prev_feature.shape[1], 1, 1).to(self.device)
                                elif output.dim() == 3: a = (torch.rand(prev_feature.shape[0], prev_feature.shape[1]) * mix_ratio + mix_lower_bound_feature).view(prev_feature.shape[0], prev_feature.shape[1], 1).to(self.device)
                                else: a = (torch.rand(prev_feature.shape[0], prev_feature.shape[1]) * mix_ratio + mix_lower_bound_feature).view(prev_feature.shape[0], prev_feature.shape[1]).to(self.device)
                            else:
                                if output.dim() == 4: a = (torch.rand(prev_feature.shape[0]) * mix_ratio + mix_lower_bound_feature).view(prev_feature.shape[0], 1, 1, 1).to(self.device)
                                elif output.dim() == 3: a = (torch.rand(prev_feature.shape[0]) * mix_ratio + mix_lower_bound_feature).view(prev_feature.shape[0], 1, 1).to(self.device)
                                else: a = (torch.rand(prev_feature.shape[0]) * mix_ratio + mix_lower_bound_feature).view(prev_feature.shape[0], 1).to(self.device)
                            if self.mixup_layer == 'relu': output = F.relu(output, inplace=True)
                            output_flat = output.detach().view(output.size(0), -1)  
                            tuning_flat = self.outputs_tuning[layer_idx].detach().view(output.size(0), -1)  
                            scale = exp_settings['ftm_beta'] * output_flat.norm(dim=1) / (tuning_flat.norm(dim=1) + 1e-7)  
                            for _ in range(len(output.shape) - 1): scale = scale.unsqueeze(-1)
                            output1 = output + self.outputs_tuning[layer_idx] * scale
                            if blending_mode_feature == 'M': return (1 - a) * output1 + a * prev_feature_shuffle
                            elif blending_mode_feature == 'A': return output1 + a * prev_feature_shuffle
                        else:
                            output_flat = output.detach().view(output.size(0), -1)  
                            tuning_flat = self.outputs_tuning[layer_idx].detach().view(output.size(0), -1)  
                            scale = exp_settings['ftm_beta'] * output_flat.norm(dim=1) / (tuning_flat.norm(dim=1) + 1e-7)  
                            for _ in range(len(output.shape) - 1): scale = scale.unsqueeze(-1)
                            return output + self.outputs_tuning[layer_idx].detach() * scale
                    elif self.record == True:  
                        cached_output = output.detach()
                        if self.cache_clean_features_cpu:
                            cached_output = cached_output.cpu().pin_memory()
                        self.outputs[layer_idx] = cached_output
                        self.outputs_tuning[layer_idx] = torch.zeros_like(output).detach().requires_grad_(True)
                        self.mixing_triggered[layer_idx] = False
        return hook_fn

    def start_feature_record(self): self.record = True
    def end_feature_record(self): self.record = False
    def remove_hooks(self):
        for fh in self.forward_hooks: fh.remove()
        self.outputs.clear(); self.outputs_tuning.clear(); self.mixing_triggered.clear()
        self.current_feature_norms.clear()
        self.forward_hooks.clear()
    def forward(self, x: Tensor) -> Tensor:
        self.mixing_triggered = {}
        return self.model(x)

class LogitLoss(nn.Module):
    def __init__(self, labels, targeted=True):
        super().__init__()
        self.labels = labels; self.targeted = targeted
    def forward(self, logits):
        real = logits.gather(1, self.labels.unsqueeze(1)).squeeze(1)
        loss = real.sum()
        return -loss if not self.targeted else loss


def model_loss_sum(models, x_step, loss_fn, compute_stats=None):
    total_loss = 0
    _count_compute(compute_stats, 'logical_attack_forward_calls')
    for model in models:
        total_loss += loss_fn(model(x_step))
        _count_compute(compute_stats, 'source_model_forward_executions')
    return total_loss


def apply_input_transform(x_step, attack_type, prob):
    if 'D' in attack_type:
        return DI(x_step, prob)
    if 'R' in attack_type:
        return RDI(x_step)
    if 'O' in attack_type:
        return ODI(x_step)
    return x_step


def calculate_si_grad(models, x_nes, attack_type, prob, loss_fn, exp_settings, compute_stats=None):
    num_scales = int(exp_settings.get('number_of_si_scales', 5))
    grad_sum = torch.zeros_like(x_nes)
    transform_attack_type = attack_type.replace('S', '')

    for scale_idx in range(num_scales):
        scale = 2 ** scale_idx
        x_neighbor = x_nes.detach().clone()
        x_neighbor.requires_grad = True
        x_scaled = scale_input_around_midpoint(x_neighbor, scale)
        x_step = apply_input_transform(x_scaled, transform_attack_type, prob)
        loss = model_loss_sum(models, x_step, loss_fn, compute_stats)
        grad = torch.autograd.grad(loss, x_neighbor, retain_graph=False, create_graph=False)[0]
        _count_compute(compute_stats, 'backward_calls')
        _count_compute(compute_stats, 'image_gradient_evaluations')
        grad_sum.add_(grad.detach())

    return grad_sum


def calculate_admix_grad(models, x_nes, clean_x, attack_type, prob, loss_fn, exp_settings, compute_stats=None):
    num_mix_samples = int(exp_settings.get('num_mix_samples', 3))
    num_scales = int(exp_settings.get('number_of_si_scales', 5))
    mix_portion = float(exp_settings.get('admix_portion', 0.2))
    grad_sum = torch.zeros_like(x_nes)
    transform_attack_type = attack_type.replace('A', '').replace('S', '')

    for _ in range(num_mix_samples):
        for scale_idx in range(num_scales):
            x_neighbor = x_nes.detach().clone()
            x_neighbor.requires_grad = True
            batch_perm = torch.randperm(clean_x.size(0), device=clean_x.device)
            mixed = x_neighbor + mix_portion * clean_x[batch_perm].detach()
            mixed = mixed.clamp(0, 1)
            scale = 2 ** scale_idx
            x_scaled = scale_input_around_midpoint(mixed, scale)
            x_step = apply_input_transform(x_scaled, transform_attack_type, prob)
            loss = model_loss_sum(models, x_step, loss_fn, compute_stats)
            grad = torch.autograd.grad(loss, x_neighbor, retain_graph=False, create_graph=False)[0]
            _count_compute(compute_stats, 'backward_calls')
            _count_compute(compute_stats, 'image_gradient_evaluations')
            grad_sum.add_(grad.detach())

    return grad_sum / max(num_mix_samples, 1)


def scale_input_around_midpoint(x, scale):
    if scale == 1:
        return x
    return ((x - 0.5) * 2.0 / scale + 1.0) / 2.0


def calculate_everywhere_grad(models, x_nes, attack_type, prob, loss_fn, exp_settings, compute_stats=None):
    grid = int(exp_settings.get('everywhere_grid', 4))
    num_samples = int(exp_settings.get('everywhere_num_samples', 9))
    grid = max(grid, 1)
    total_blocks = grid * grid
    num_samples = max(0, min(num_samples, total_blocks))
    transform_attack_type = attack_type.replace('E', '')
    grad_sum = torch.zeros_like(x_nes)

    def one_grad(mask=None):
        x_neighbor = x_nes.detach().clone()
        x_neighbor.requires_grad = True
        x_step = apply_input_transform(x_neighbor, transform_attack_type, prob)
        if mask is not None:
            x_step = apply_everywhere_mask(x_step, mask, exp_settings)
        loss = model_loss_sum(models, x_step, loss_fn, compute_stats)
        grad = torch.autograd.grad(loss, x_neighbor, retain_graph=False, create_graph=False)[0]
        _count_compute(compute_stats, 'backward_calls')
        _count_compute(compute_stats, 'image_gradient_evaluations')
        return grad

    grad_sum.add_(one_grad().detach())
    if num_samples > 0:
        block_indices = torch.randperm(total_blocks, device=x_nes.device)[:num_samples]
        for block_idx in block_indices.tolist():
            mask = make_everywhere_mask(x_nes, grid, block_idx)
            grad_sum.add_(one_grad(mask).detach())

    return grad_sum / (num_samples + 1)


def make_everywhere_mask(x, grid, block_idx):
    _, _, height, width = x.shape
    row = block_idx // grid
    col = block_idx % grid
    top = int(np.floor(row * height / grid))
    bottom = int(np.floor((row + 1) * height / grid))
    left = int(np.floor(col * width / grid))
    right = int(np.floor((col + 1) * width / grid))
    mask = torch.zeros(1, 1, height, width, device=x.device, dtype=x.dtype)
    mask[:, :, top:bottom, left:right] = 1
    return mask


def apply_everywhere_mask(x, mask, exp_settings):
    fill_mode = exp_settings.get('everywhere_fill', 'mean')
    if fill_mode == 'zero':
        return x * mask
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x * mask + mean * (1 - mask)


def DI(X_in, prob):
    img_width = X_in.size()[-1]  
    enlarged_img_width = int(img_width * 330. / 299.)
    rnd = np.random.randint(img_width, enlarged_img_width, size=1)[0]
    h_rem = enlarged_img_width - rnd; w_rem = enlarged_img_width - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]; pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]; pad_right = w_rem - pad_left
    if np.random.rand() <= prob: return F.pad(F.interpolate(X_in, size=(rnd, rnd), mode='nearest'), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return X_in

def RDI(x_adv):
    img_width = x_adv.size()[-1]
    enlarged_img_width = int(img_width * 340. / 299.)
    di_pad_amount = enlarged_img_width - img_width
    ori_size = x_adv.shape[-1]
    rnd = int(torch.rand(1).item() * di_pad_amount) + ori_size
    x_di = F.interpolate(x_adv, size=(rnd, rnd), mode='nearest')
    pad_max = ori_size + di_pad_amount - rnd
    pad_left = int(torch.rand(1).item() * max(pad_max, 1)); pad_right = max(pad_max - pad_left, 0)
    pad_top = int(torch.rand(1).item() * max(pad_max, 1)); pad_bottom = max(pad_max - pad_top, 0)
    x_di = F.pad(x_di, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
    if img_width > 64: x_di = F.interpolate(x_di, size=(ori_size, ori_size), mode='nearest')
    return x_di

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    return kernel_raw / kernel_raw.sum()

def SU_augment_single(x, scale_start=0.5, scale_interval=0.2):
    B, C, H, W = x.shape
    random_scale = torch.rand(1, device=x.device).item() * scale_interval + scale_start
    crop_h, crop_w = max(int(H * random_scale), 10), max(int(W * random_scale), 10)
    top, left = (H - crop_h) // 2, (W - crop_w) // 2
    return F.interpolate(x[:, :, top:top+crop_h, left:left+crop_w], size=(H, W), mode='bilinear', align_corners=False)


class CleanFeatureMixup(nn.Module):
    def __init__(self, model: nn.Module, input_size, exp_settings, device):
        super().__init__()
        self.exp_settings = exp_settings
        self.device = device
        self.mixup_layer = exp_settings['mixup_layer']
        self.prob = exp_settings['mix_prob']
        self.channelwise = exp_settings['channelwise']
        self.model = model
        self.input_size = input_size
        self.record = False
        self.outputs = {}
        self.forward_hooks = []

        def get_children(module: torch.nn.Module):
            children = list(module.children())
            flattened_children = []
            if children == []:
                if self.mixup_layer in ('conv_linear_no_last', 'conv_linear_include_last'):
                    if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
                        return module
                    return []
                if self.mixup_layer == 'bn' or self.mixup_layer == 'relu':
                    if type(module) == torch.nn.BatchNorm2d:
                        return module
                    return []
                if type(module) == torch.nn.Conv2d:
                    return module
                return []
            for child in children:
                try:
                    flattened_children.extend(get_children(child))
                except TypeError:
                    flattened_children.append(get_children(child))
            return flattened_children

        mod_list = get_children(model)
        self.layer_num = len(mod_list)
        for i, module in enumerate(mod_list):
            self.forward_hooks.append(module.register_forward_hook(self.save_outputs_hook(i)))

    def save_outputs_hook(self, layer_idx) -> Callable:
        exp_settings = self.exp_settings
        mix_upper_bound_feature = exp_settings['mix_upper_bound_feature']
        mix_lower_bound_feature = exp_settings['mix_lower_bound_feature']
        shuffle_image_feature = exp_settings['shuffle_image_feature']
        blending_mode_feature = exp_settings['blending_mode_feature']
        mixed_image_type_feature = exp_settings['mixed_image_type_feature']
        divisor = exp_settings['divisor']

        def hook_fn(module, input, output):
            if type(module) == torch.nn.Linear or output.size()[-1] <= self.input_size // divisor:
                if self.mixup_layer == 'conv_linear_no_last' and (layer_idx + 1) == self.layer_num and type(module) == torch.nn.Linear:
                    return None
                if layer_idx in self.outputs and self.record == False:
                    if torch.rand(1).item() > self.prob:
                        return output
                    if mixed_image_type_feature == 'A':
                        prev_feature = output.clone().detach()
                    else:
                        prev_feature = self.outputs[layer_idx].clone().detach()
                    if shuffle_image_feature == 'SelfShuffle':
                        idx = torch.randperm(output.shape[0], device=output.device)
                        prev_feature_shuffle = prev_feature[idx].view(prev_feature.size())
                    else:
                        prev_feature_shuffle = prev_feature

                    mix_ratio = mix_upper_bound_feature - mix_lower_bound_feature
                    if self.channelwise:
                        if output.dim() == 4:
                            shape = (prev_feature.shape[0], prev_feature.shape[1], 1, 1)
                        elif output.dim() == 3:
                            shape = (prev_feature.shape[0], prev_feature.shape[1], 1)
                        else:
                            shape = (prev_feature.shape[0], prev_feature.shape[1])
                        a = (torch.rand(shape[:2], device=self.device) * mix_ratio + mix_lower_bound_feature).view(shape)
                    else:
                        if output.dim() == 4:
                            shape = (prev_feature.shape[0], 1, 1, 1)
                        elif output.dim() == 3:
                            shape = (prev_feature.shape[0], 1, 1)
                        else:
                            shape = (prev_feature.shape[0], 1)
                        a = (torch.rand(prev_feature.shape[0], device=self.device) * mix_ratio + mix_lower_bound_feature).view(shape)

                    if self.mixup_layer == 'relu':
                        output = F.relu(output, inplace=True)
                    if blending_mode_feature == 'M':
                        return (1 - a) * output + a * prev_feature_shuffle
                    if blending_mode_feature == 'A':
                        return output + a * prev_feature_shuffle
                    return output
                if self.record == True:
                    self.outputs[layer_idx] = output.clone().detach()
            return None
        return hook_fn

    def start_feature_record(self):
        self.record = True

    def end_feature_record(self):
        self.record = False

    def remove_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.outputs.clear()
        self.forward_hooks.clear()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def ODI(x_adv, resize_ratio=330.0/299.0):
    img_width = x_adv.size()[-1]
    enlarged_img_width = int(img_width * resize_ratio)
    rnd = np.random.randint(img_width, enlarged_img_width + 1, size=1)[0]
    h_rem = w_rem = enlarged_img_width - rnd
    pad_top = np.random.randint(0, max(h_rem, 1), size=1)[0]; pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, max(w_rem, 1), size=1)[0]; pad_right = w_rem - pad_left
    x_out = F.pad(F.interpolate(x_adv, size=(rnd, rnd), mode='nearest'), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return F.interpolate(x_out, size=(img_width, img_width), mode='nearest')

class CFM(nn.Module):
    def __init__(self, model: nn.Module, device, num_layers=3, mix_beta=0.01):
        super().__init__()
        self.model, self.device, self.num_layers, self.mix_beta = model, device, num_layers, mix_beta
        self.hooks, self.layer_names, self.prev_feature = [], [], None
        layer_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and layer_count < self.num_layers:
                layer_count += 1; self.layer_names.append(name)
                def make_hook(layer_name):
                    def hook_fn(module, input, output):
                        if self.prev_feature is not None and len(output.shape) == 4 and len(self.prev_feature.shape) == 4:
                            if self.prev_feature.shape[1] == output.shape[1]:
                                alpha = torch.rand(1, device=self.device).item() * self.mix_beta
                                prev_resized = F.interpolate(self.prev_feature, size=output.shape[-2:], mode='nearest') if self.prev_feature.shape[2:] != output.shape[2:] else self.prev_feature
                                output = output + alpha * prev_resized.detach()
                        self.prev_feature = output.detach().clone()
                        return output
                    return hook_fn
                self.hooks.append(module.register_forward_hook(make_hook(name)))
    def remove_hooks(self):
        for hook in self.hooks: hook.remove()
        self.hooks.clear()
        self.prev_feature = None
    def forward(self, x):
        self.prev_feature = None
        return self.model(x)
