import os
import csv
import hashlib
from urllib.parse import urlparse
import numpy as np
import PIL.Image as Image
from collections import OrderedDict

import torchvision
import torch.nn as nn
import torch
import timm
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F


LOCAL_VIT_MODEL_ARCH_AND_FILE = OrderedDict([
    ("vit_base_patch16_224", ("vit_base_patch16_224", "vit_base_patch16_224.bin")),
    ("deit_tiny_patch16_224", ("deit_tiny_patch16_224", "deit_tiny_patch16_224.bin")),
    # This local file contains distilled checkpoints in this workspace.
    ("deit_small_patch16_224", ("deit_small_distilled_patch16_224", "deit_small_patch16_224.bin")),
    ("deit_base_distilled_patch16_224", ("deit_base_distilled_patch16_224", "deit_base_distilled_patch16_224.bin")),
    ("swin_tiny_patch4_window7_224", ("swin_tiny_patch4_window7_224", "swin_tiny_patch4_window7_224.bin")),
    ("swin_small_patch4_window7_224", ("swin_small_patch4_window7_224", "swin_small_patch4_window7_224.bin")),
    ("cait_s24_224", ("cait_s24_224", "cait_s24_224.bin")),
    ("coat_tiny", ("coat_tiny", "coat_tiny.bin")),
    ("visformer_tiny", ("visformer_tiny", "visformer_tiny.bin")),
    ("visformer_small", ("visformer_small", "visformer_small.bin")),
    ("pit_ti_224", ("pit_ti_224", "pit_ti_224.bin")),
    ("pit_b_224", ("pit_b_224", "pit_b_224.bin")),
])

LOCAL_VIT_MODEL_NAMES = list(LOCAL_VIT_MODEL_ARCH_AND_FILE.keys())


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretrained_weight_info(model_name, model):
    builder_name = {"vgg16": "vgg16_bn", "vgg19": "vgg19_bn"}.get(
        model_name, model_name.lower()
    )
    builder = (getattr(torchvision.models, builder_name, None)
               if model.__class__.__module__.startswith("torchvision.") else None)
    identifier = None
    if builder is not None:
        try:
            # The legacy pretrained=True API resolves to the first weight enum.
            identifier = list(torchvision.models.get_model_weights(builder))[0].url
        except (AttributeError, IndexError, ValueError):
            pass
    if identifier is None:
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        identifier = cfg.get("url") or cfg.get("hf_hub_id")
    checkpoint_path = None
    if identifier and str(identifier).startswith(("http://", "https://")):
        candidate = os.path.join(torch.hub.get_dir(), "checkpoints",
                                 os.path.basename(urlparse(identifier).path))
        if os.path.isfile(candidate):
            checkpoint_path = os.path.abspath(candidate)
    return {
        "source": "pretrained",
        "weight_identifier": identifier,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": _file_sha256(checkpoint_path) if checkpoint_path else None,
        "loaded_tensor_count": None,
        "missing_keys": None,
        "unexpected_keys": None,
        "shape_mismatched_keys": None,
    }


def _unwrap_state_dict_if_needed(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                ckpt_obj = ckpt_obj[key]
                break

    if not isinstance(ckpt_obj, dict):
        raise ValueError("Unsupported checkpoint format. Expected a state_dict-like object.")

    state_dict = ckpt_obj
    if len(state_dict) > 0 and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _try_load_local_timm_weight(model_name, model_weights_dir):
    if not model_weights_dir:
        return None
    if model_name not in LOCAL_VIT_MODEL_ARCH_AND_FILE:
        return None

    timm_arch, weight_file = LOCAL_VIT_MODEL_ARCH_AND_FILE[model_name]
    weight_path = os.path.join(model_weights_dir, weight_file)
    if not os.path.isfile(weight_path):
        return None

    model = timm.create_model(timm_arch, pretrained=False)
    try:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = _unwrap_state_dict_if_needed(checkpoint)

    # Keep only tensors with matching keys and shapes to tolerate classifier-head mismatches.
    model_state = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    if len(filtered_state_dict) == 0:
        raise RuntimeError(
            f"No compatible tensors found when loading local weight: {weight_path}"
        )

    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    model._codex_weight_info = {
        "source": "local",
        "weight_identifier": timm_arch,
        "checkpoint_path": os.path.abspath(weight_path),
        "checkpoint_sha256": _file_sha256(weight_path),
        "loaded_tensor_count": len(filtered_state_dict),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": [k for k in state_dict if k not in model_state],
        "shape_mismatched_keys": [k for k, v in state_dict.items()
                                  if k in model_state and model_state[k].shape != v.shape],
    }
    return model


def get_available_local_vit_models(model_weights_dir):
    if not model_weights_dir or not os.path.isdir(model_weights_dir):
        return []

    model_names = []
    for model_name, (_, weight_file) in LOCAL_VIT_MODEL_ARCH_AND_FILE.items():
        if os.path.isfile(os.path.join(model_weights_dir, weight_file)):
            model_names.append(model_name)
    return model_names


def _normalize_to_zero_based(labels):
    if len(labels) == 0:
        return labels
    if min(labels) >= 1:
        return [x - 1 for x in labels]
    return labels


# simple wrapper model to normalize an input image
class WrapperModel(nn.Module):
    def __init__(self, model, mean, std, resize=False):
        super(WrapperModel, self).__init__()
        self.mean = torch.Tensor(mean)
        self.model = model
        self.resize = resize
        self.std = torch.Tensor(std)

    def forward(self, x):
        if self.resize == True:
            x = transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)(x)
        return self.model((x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None])

    def normalize(self, x):
        if self.resize == True:
            x = transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)(x)
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


def load_model(model_name, model_weights_dir="", prefer_local_weights=False):
    model = _try_load_local_timm_weight(model_name, model_weights_dir)
    if model is not None:
        return model

    if prefer_local_weights and model_name in LOCAL_VIT_MODEL_ARCH_AND_FILE:
        _, weight_file = LOCAL_VIT_MODEL_ARCH_AND_FILE[model_name]
        expected_path = os.path.join(model_weights_dir, weight_file)
        raise FileNotFoundError(
            f"Local weight for {model_name} not found at: {expected_path}"
        )

    if model_name == "ResNet101":
        model = torchvision.models.resnet101(pretrained=True)
    elif model_name == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'ResNet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif model_name == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "ResNet152":
        model = torchvision.models.resnet152(pretrained=True)
    elif model_name == "vgg16":
        model = torchvision.models.vgg16_bn(pretrained=True)
    elif model_name == "vgg19":
        model = torchvision.models.vgg19_bn(pretrained=True)
    elif model_name == "wide_resnet101_2":
        model = torchvision.models.wide_resnet101_2(pretrained=True)
    elif model_name == "inception_v3":
        model = torchvision.models.inception_v3(pretrained=True, transform_input=True)
    elif model_name == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(pretrained=True)
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
    elif model_name == "mobilenet_v3_large":
        model = torchvision.models.mobilenet.mobilenet_v3_large(pretrained=True)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
    elif model_name == "DenseNet161":
        model = torchvision.models.densenet161(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
    elif model_name == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == 'GoogLeNet':
        model = torchvision.models.googlenet(pretrained=True)
    # timm models
    elif model_name == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True)
    elif model_name == "inception_resnet_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=True)
    elif model_name == "inception_v3_timm":
        model = timm.create_model("inception_v3", pretrained=True)
    elif model_name == "inception_v4_timm":
        model = timm.create_model("inception_v4", pretrained=True)
    elif model_name == "xception":
        model = timm.create_model("xception", pretrained=True)
    # timm Transformer-based models
    elif model_name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
    elif model_name == "levit_384":
        model = timm.create_model("levit_384", pretrained=True)
    elif model_name == "convit_base":
        model = timm.create_model("convit_base", pretrained=True)
    elif model_name == "twins_svt_base":
        model = timm.create_model("twins_svt_base", pretrained=True)
    elif model_name == "pit":
        model = timm.create_model('pit_s_224', pretrained=True)
    elif model_name == "deit_tiny_patch16_224":
        model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    elif model_name == "deit_small_patch16_224":
        model = timm.create_model("deit_small_distilled_patch16_224", pretrained=True)
    elif model_name == "deit_base_distilled_patch16_224":
        model = timm.create_model("deit_base_distilled_patch16_224", pretrained=True)
    elif model_name == "swin_tiny_patch4_window7_224":
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
    elif model_name == "swin_small_patch4_window7_224":
        model = timm.create_model("swin_small_patch4_window7_224", pretrained=True)
    elif model_name == "cait_s24_224":
        model = timm.create_model("cait_s24_224", pretrained=True)
    elif model_name == "coat_tiny":
        model = timm.create_model("coat_tiny", pretrained=True)
    elif model_name == "visformer_tiny":
        model = timm.create_model("visformer_tiny", pretrained=True)
    elif model_name == "visformer_small":
        model = timm.create_model("visformer_small", pretrained=True)
    elif model_name == "pit_ti_224":
        model = timm.create_model("pit_ti_224", pretrained=True)
    elif model_name == "pit_b_224":
        model = timm.create_model("pit_b_224", pretrained=True)
    else:
        raise ValueError(f"Not supported model name. {model_name}")
    model._codex_weight_info = _pretrained_weight_info(model_name, model)
    return model


# load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename, return_metadata=False):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        fieldnames = set(reader.fieldnames or [])

        # Original FTM format.
        if {'ImageId', 'TrueLabel', 'TargetClass'}.issubset(fieldnames):
            for row in reader:
                image_id_list.append(row['ImageId'].strip())
                label_ori_list.append(int(row['TrueLabel']))
                label_tar_list.append(int(row['TargetClass']))
        # FPR dataset format.
        elif {'filename', 'label', 'targeted_label'}.issubset(fieldnames):
            for row in reader:
                image_id_list.append(row['filename'].strip())
                label_ori_list.append(int(row['label']))
                label_tar_list.append(int(row['targeted_label']))
        else:
            raise ValueError(
                f"Unsupported CSV format in {csv_filename}. "
                "Expected columns: [ImageId, TrueLabel, TargetClass] "
                "or [filename, label, targeted_label]."
            )

    raw_ori_list = list(label_ori_list)
    raw_tar_list = list(label_tar_list)
    # Preserve the historical minimum-based label normalization.
    ori_shifted = bool(label_ori_list and min(label_ori_list) >= 1)
    tar_shifted = bool(label_tar_list and min(label_tar_list) >= 1)
    label_ori_list = _normalize_to_zero_based(label_ori_list)
    label_tar_list = _normalize_to_zero_based(label_tar_list)

    if return_metadata:
        return image_id_list, label_ori_list, label_tar_list, {
            "raw_true_labels": raw_ori_list,
            "raw_target_labels": raw_tar_list,
            "true_label_shifted": ori_shifted,
            "target_label_shifted": tar_shifted,
        }
    return image_id_list, label_ori_list, label_tar_list


class EvalResult:
    def __init__(self, model_name):
        self.model_name = model_name
        self.total_samples = 0
        self.success_samples = 0
        self.success_rate = 0.0

    def update(self, pred_labels, target_labels):
        batch_size = len(pred_labels)
        self.total_samples += batch_size
        self.success_samples += torch.sum(pred_labels == target_labels).item()
        self.success_rate = self.success_samples / self.total_samples * 100

    def get_results(self):
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "success_samples": self.success_samples,
            "success_rate": self.success_rate
        }

    def __str__(self):
        return f"Model: {self.model_name:<21} | Attack Success: {self.success_rate:.1f}% ({self.success_samples}/{self.total_samples})"


def save_images(img_tensor, save_path):
    img = np.array(img_tensor.cpu().detach().numpy()).transpose(1, 2, 0) * 255.
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    im.save(save_path)
