"""Factory contract for the 3 frozen architectures. Builds with
pretrained=False (no download) and checks head + group wiring.
"""

import torch

from src.models.factory import (
    ARCHITECTURES,
    _group_containers,
    apply_unfreeze,
    build_model,
    head_module,
)


def test_registry_is_exactly_the_frozen_three():
    assert set(ARCHITECTURES) == {"mobilenetv3_small_100", "resnet18", "mobilevit_xxs"}


def test_each_arch_builds_6class_head():
    for arch in ARCHITECTURES:
        model = build_model(arch, num_classes=6, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        assert model(x).shape == (1, 6)


def test_frozen_then_unfrozen_param_counts():
    for arch in ARCHITECTURES:
        model = build_model(arch, num_classes=6, pretrained=False)
        apply_unfreeze(model, arch, unfreeze_backbone=False)
        n_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        apply_unfreeze(model, arch, unfreeze_backbone=True)
        n_full = sum(p.numel() for p in model.parameters() if p.requires_grad)
        head_n = sum(p.numel() for p in head_module(model, arch).parameters())
        assert n_frozen == head_n
        assert n_full >= n_frozen
        assert len(_group_containers(model, arch)) >= 3
