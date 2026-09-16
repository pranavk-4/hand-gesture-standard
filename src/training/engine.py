"""Per-epoch train/eval loops. Verbatim port of
hand_gesture_classifier/src/training/engine.py (grad-norm capture by
group, wall-clock timing, confidences for ECE). No changes needed for
the 6-class cut — it is class-count agnostic.
"""

import math
import time

import torch
import torch.nn.functional as F

from src.training.metrics import gradient_norms_by_group


def train_one_epoch(model, loader, optimizer, criterion, device, head_params, use_amp: bool,
                    grad_clip_norm: float = 0.0) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    grad_norm_sum = {"head": 0.0, "backbone": 0.0, "total": 0.0}
    grad_norm_count = {"head": 0, "backbone": 0, "total": 0}
    start = time.time()

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        batch_norms = gradient_norms_by_group(model, head_params)
        for key in grad_norm_sum:
            value = batch_norms[key]
            if math.isfinite(value):
                grad_norm_sum[key] += value
                grad_norm_count[key] += 1

        # Clip AFTER unscale (grad norms above are the pre-clip diagnostic).
        # Stops the pretrained backbone from being blown out on small data.
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "grad_norms": {k: (grad_norm_sum[k] / grad_norm_count[k] if grad_norm_count[k] > 0 else float("nan"))
                       for k in grad_norm_sum},
        "epoch_seconds": time.time() - start,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_confidences = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "predictions": all_preds,
        "labels": all_labels,
        "confidences": all_confidences,
    }
