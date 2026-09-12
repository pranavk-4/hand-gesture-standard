"""Builds any of the 3 standard architectures through one uniform interface.

Slimmed from hand_gesture_classifier/src/models/factory.py (5 archs) to
the 3 fast-iteration models the team froze:
  - mobilenetv3_small_100 (edge CNN, primary baseline)
  - resnet18              (universal reference baseline)
  - mobilevit_xxs         (light hybrid ViT; note _xxs, not the old _xs)

Unfreeze semantics are unchanged: _group_containers() excludes the stem
and head-adjacent layers; apply_unfreeze(True) unfreezes every group.
"""

import timm
import torch.nn as nn

ARCHITECTURES = (
    "mobilenetv3_small_100",
    "resnet18",
    "mobilevit_xxs",
)


def build_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}, expected one of {ARCHITECTURES}")
    return timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)


def _group_containers(model: nn.Module, architecture: str) -> list[nn.Module]:
    if architecture == "resnet18":
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    if architecture == "mobilenetv3_small_100":
        return list(model.blocks)
    if architecture == "mobilevit_xxs":
        return list(model.stages)
    raise ValueError(f"No group-container mapping for {architecture!r}")


def head_module(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "mobilenetv3_small_100":
        return model.classifier
    if architecture == "resnet18":
        return model.fc
    if architecture == "mobilevit_xxs":
        return model.head
    raise ValueError(f"No head mapping for {architecture!r}")


def apply_unfreeze(model: nn.Module, architecture: str, unfreeze_backbone: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_backbone:
        for group in _group_containers(model, architecture):
            for param in group.parameters():
                param.requires_grad = True

    for param in head_module(model, architecture).parameters():
        param.requires_grad = True


def trainable_parameter_groups(model: nn.Module, architecture: str, head_lr: float,
                               backbone_lr: float, differential_lr: bool) -> list[dict]:
    head_params = list(head_module(model, architecture).parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]

    if not differential_lr:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": head_lr}]

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": [p for p in head_params if p.requires_grad], "lr": head_lr},
    ]
