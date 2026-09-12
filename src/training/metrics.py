"""Gradient-norm, generalization-gap, and ECE metrics. Verbatim port of
hand_gesture_classifier/src/training/metrics.py.
"""

import torch
import torch.nn as nn


def gradient_norms_by_group(model: nn.Module, head_params: list[torch.nn.Parameter]) -> dict:
    head_ids = {id(p) for p in head_params}
    head_sq_sum, backbone_sq_sum = 0.0, 0.0

    for param in model.parameters():
        if param.grad is None:
            continue
        sq_sum = float(param.grad.data.norm(2) ** 2)
        if id(param) in head_ids:
            head_sq_sum += sq_sum
        else:
            backbone_sq_sum += sq_sum

    return {
        "head": head_sq_sum ** 0.5,
        "backbone": backbone_sq_sum ** 0.5,
        "total": (head_sq_sum + backbone_sq_sum) ** 0.5,
    }


def generalization_gap(train_acc: float, val_acc: float) -> float:
    return train_acc - val_acc


def expected_calibration_error(confidences: list[float], correct: list[bool], num_bins: int = 10) -> float:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not confidences:
        return 0.0

    bin_edges = [i / num_bins for i in range(num_bins + 1)]
    total = len(confidences)
    ece = 0.0

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [
            (c, ok) for c, ok in zip(confidences, correct)
            if (lo <= c < hi) or (i == num_bins - 1 and c == hi)
        ]
        if not in_bin:
            continue
        bin_confidence = sum(c for c, _ in in_bin) / len(in_bin)
        bin_accuracy = sum(ok for _, ok in in_bin) / len(in_bin)
        ece += (len(in_bin) / total) * abs(bin_confidence - bin_accuracy)

    return ece
