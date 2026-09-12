"""Exponential moving average of model weights. Verbatim port of
hand_gesture_classifier/src/training/ema.py (architecture-agnostic).
"""

import copy

import torch
import torch.nn as nn


class WeightEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow_model = copy.deepcopy(model)
        for param in self.shadow_model.parameters():
            param.requires_grad = False
        self.shadow_model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for shadow_param, current_param in zip(
            self.shadow_model.state_dict().values(), model.state_dict().values()
        ):
            if shadow_param.dtype.is_floating_point:
                shadow_param.mul_(self.decay).add_(current_param, alpha=1 - self.decay)
            else:
                shadow_param.copy_(current_param)

    def state_dict(self):
        return self.shadow_model.state_dict()
