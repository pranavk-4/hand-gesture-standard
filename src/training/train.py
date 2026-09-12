"""Baseline training entrypoint: ONE config -> ONE run. No ablation stages.

Replaces hand_gesture_classifier/src/training/train.py (which required a
`stage` int and resolved AblationSettings). All technique flags now come
from `cfg.training` directly, with optional per-call `overrides` dict so
the HPO track can try hyperparams without editing configs:

    run_training(cfg, run_name="baseline", overrides={"head_lr": 3e-4})

Outputs (same JSON contract as the research pipeline so evaluate/plots
keep working):
  <checkpoints_dir>/<run_name>/best_model.pth
  <metrics_dir>/<run_name>/training_history.json
  <metrics_dir>/<run_name>/training_summary.json
"""

import json
import logging
import time

import torch
import torch.nn as nn

from src.data.datasets import build_dataloaders
from src.models.factory import apply_unfreeze, build_model, trainable_parameter_groups
from src.training.ema import WeightEMA
from src.training.engine import evaluate, train_one_epoch

logger = logging.getLogger(__name__)


def _resolve_flag(cfg, overrides: dict | None, name: str):
    if overrides and name in overrides:
        return overrides[name]
    return getattr(cfg.training, name)


def build_optimizer_and_scheduler(model, cfg, architecture: str, overrides: dict | None = None):
    head_lr = _resolve_flag(cfg, overrides, "head_lr")
    backbone_lr = _resolve_flag(cfg, overrides, "backbone_lr")
    differential_lr = _resolve_flag(cfg, overrides, "differential_lr")
    cosine_schedule = _resolve_flag(cfg, overrides, "cosine_schedule")
    warmup = _resolve_flag(cfg, overrides, "warmup")

    param_groups = trainable_parameter_groups(model, architecture, head_lr, backbone_lr, differential_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=_resolve_flag(cfg, overrides, "weight_decay"))

    if not cosine_schedule:
        return optimizer, None

    warmup_epochs = cfg.training.warmup_epochs if warmup else 0
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.max_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.max_epochs)
    return optimizer, scheduler


def run_training(cfg, run_name: str, overrides: dict | None = None,
                 max_epochs_override: int | None = None) -> dict:
    """Train one baseline run. `overrides` may contain any cfg.training key
    (head_lr, backbone_lr, weight_decay, ema_decay, label_smoothing_value,
    unfreeze_backbone, warmup, cosine_schedule, class_weighted_loss,
    label_smoothing, ema, differential_lr, augmentation).

    `max_epochs_override` is the HPO multi-fidelity budget (Hyperband
    bracket resource). When set, early-stopping patience scales down
    proportionally so low-budget trials can still stop early.
    """
    overrides = overrides or {}
    architecture = cfg.model.architecture
    max_epochs = max_epochs_override or cfg.training.max_epochs

    checkpoints_dir = cfg.output.checkpoints_dir / run_name
    metrics_dir = cfg.output.metrics_dir / run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Run %s | architecture %s | device %s | overrides %s",
                run_name, architecture, device, overrides or "none")

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        cfg,
        use_heavy_augmentation=_resolve_flag(cfg, overrides, "augmentation"),
        use_class_weights=_resolve_flag(cfg, overrides, "class_weighted_loss"),
    )

    model = build_model(architecture, num_classes=len(cfg.dataset.class_names),
                        pretrained=cfg.model.pretrained)
    apply_unfreeze(model, architecture, _resolve_flag(cfg, overrides, "unfreeze_backbone"))
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    label_smoothing = cfg.training.label_smoothing_value if _resolve_flag(cfg, overrides, "label_smoothing") else 0.0
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, architecture, overrides)
    use_ema = _resolve_flag(cfg, overrides, "ema")
    ema = WeightEMA(model, _resolve_flag(cfg, overrides, "ema_decay")) if use_ema else None

    from src.models.factory import head_module
    head_params = list(head_module(model, architecture).parameters())

    patience = cfg.training.early_stopping_patience
    if max_epochs_override is not None:
        patience = max(2, round(patience * max_epochs_override / cfg.training.max_epochs))

    best_val_acc, best_state, best_is_ema = 0.0, None, False
    epochs_without_improvement = 0
    history = []
    run_start = time.time()

    for epoch in range(1, max_epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                       head_params, use_amp=cfg.training.use_amp)
        val_result = evaluate(model, val_loader, criterion, device)

        ema_val_acc = None
        if ema is not None:
            ema.update(model)
            ema_val_acc = evaluate(ema.shadow_model, val_loader, criterion, device)["accuracy"]

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[-1]["lr"]

        gap = train_result["accuracy"] - val_result["accuracy"]
        logger.info(
            "epoch %02d | train_loss %.4f train_acc %.4f | val_loss %.4f val_acc %.4f | "
            "gap %.4f | grad_norm(head=%.3f backbone=%.3f) | lr %.6f | %.1fs",
            epoch, train_result["loss"], train_result["accuracy"], val_result["loss"], val_result["accuracy"],
            gap, train_result["grad_norms"]["head"], train_result["grad_norms"]["backbone"],
            current_lr, train_result["epoch_seconds"],
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_result["loss"], "train_accuracy": train_result["accuracy"],
            "val_loss": val_result["loss"], "val_accuracy": val_result["accuracy"],
            "ema_val_accuracy": ema_val_acc,
            "learning_rate": current_lr,
            "grad_norm_head": train_result["grad_norms"]["head"],
            "grad_norm_backbone": train_result["grad_norms"]["backbone"],
            "grad_norm_total": train_result["grad_norms"]["total"],
            "generalization_gap": gap,
            "epoch_seconds": train_result["epoch_seconds"],
        })
        with open(metrics_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        candidate_acc, candidate_is_ema = val_result["accuracy"], False
        if ema_val_acc is not None and ema_val_acc > candidate_acc:
            candidate_acc, candidate_is_ema = ema_val_acc, True

        if candidate_acc > best_val_acc:
            epochs_without_improvement = 0
            best_val_acc, best_is_ema = candidate_acc, candidate_is_ema
            source_model = ema.shadow_model if candidate_is_ema else model
            best_state = {k: v.cpu().clone() for k, v in source_model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "epoch": epoch, "val_accuracy": best_val_acc,
                        "is_ema": best_is_ema, "overrides": overrides}, checkpoints_dir / "best_model.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d.", epoch)
            break

    model.load_state_dict(best_state)
    test_result = evaluate(model, test_loader, criterion, device)

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    param_count = sum(p.numel() for p in model.parameters())

    summary = {
        "run_name": run_name, "architecture": architecture,
        "overrides": overrides, "max_epochs_budget": max_epochs,
        "best_val_accuracy": best_val_acc, "best_checkpoint_source": "ema" if best_is_ema else "raw",
        "test_accuracy": test_result["accuracy"], "test_loss": test_result["loss"],
        "test_predictions": test_result["predictions"], "test_labels": test_result["labels"],
        "test_confidences": test_result["confidences"],
        "class_names": list(cfg.dataset.class_names),
        "epochs_trained": len(history), "best_epoch": max(history, key=lambda h: h["val_accuracy"])["epoch"],
        "total_wall_clock_seconds": time.time() - run_start,
        "peak_gpu_memory_mb": peak_memory_mb,
        "param_count": param_count,
    }
    with open(metrics_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
