"""Training/evaluation plots. Simplified from
hand_gesture_classifier/src/viz/plots.py: single-run plots only
(loss, accuracy, LR, confusion matrix, per-class bars, class
distribution). The cross-stage ablation charts are dropped — the
standard pipeline trains ONE baseline per model, and cross-trial
charts belong to the HPO track.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
COLOR_TARGET = "#d03b3b"
COLOR_GRID = "#e1e0d9"
COLOR_EMA = "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.8,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def _load_json(directory: Path, name: str):
    with open(Path(directory) / name, encoding="utf-8") as f:
        return json.load(f)


def _save(fig, plots_dir: Path, filename: str) -> None:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / filename
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_loss_curve(metrics_dir: Path, plots_dir: Path) -> None:
    history = _load_json(metrics_dir, "training_history.json")
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [h["train_loss"] for h in history], color=COLOR_TRAIN, linewidth=2, label="Train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], color=COLOR_VAL, linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.legend(frameon=False); fig.tight_layout()
    _save(fig, plots_dir, "loss_curve.png")


def plot_accuracy_curve(metrics_dir: Path, plots_dir: Path) -> None:
    history = _load_json(metrics_dir, "training_history.json")
    epochs = [h["epoch"] for h in history]
    val_accs = [h["val_accuracy"] for h in history]
    ema_val_accs = [h.get("ema_val_accuracy") for h in history]
    has_ema = any(v is not None for v in ema_val_accs)
    best_acc = max(val_accs)
    best_epoch = epochs[int(np.argmax(val_accs))]
    if has_ema:
        ema_values = [v for v in ema_val_accs if v is not None]
        best_ema = max(ema_values)
        if best_ema > best_acc:
            best_acc = best_ema
            best_epoch = epochs[max(i for i, v in enumerate(ema_val_accs) if v == best_ema)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [h["train_accuracy"] for h in history], color=COLOR_TRAIN, linewidth=2, label="Train accuracy")
    ax.plot(epochs, val_accs, color=COLOR_VAL, linewidth=2, label="Val accuracy")
    if has_ema:
        ax.plot(epochs, ema_val_accs, color=COLOR_EMA, linewidth=2, linestyle="--", label="EMA val accuracy")
    ax.scatter([best_epoch], [best_acc], color=COLOR_VAL, s=70, zorder=5, edgecolor="white")
    ax.annotate(f"best: {best_acc:.1%}", (best_epoch, best_acc),
                textcoords="offset points", xytext=(8, -14), fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("Training and validation accuracy")
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout()
    _save(fig, plots_dir, "accuracy_curve.png")


def plot_learning_rate_schedule(metrics_dir: Path, plots_dir: Path) -> None:
    history = _load_json(metrics_dir, "training_history.json")
    epochs = [h["epoch"] for h in history]
    lrs = [h["learning_rate"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(epochs, lrs, color=COLOR_TRAIN, linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedule"); fig.tight_layout()
    _save(fig, plots_dir, "learning_rate_schedule.png")


def plot_confusion_matrix(metrics_dir: Path, plots_dir: Path) -> None:
    evaluation = _load_json(metrics_dir, "evaluation_report.json")
    cm = np.array(evaluation["confusion_matrix"])
    class_names = evaluation["class_names"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix (test accuracy {evaluation['test_accuracy']:.1%})")
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm[i, j]
            color = "white" if value > threshold else "#17151c"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Number of images")
    fig.tight_layout()
    _save(fig, plots_dir, "confusion_matrix.png")


def plot_per_class_metrics(metrics_dir: Path, plots_dir: Path) -> None:
    evaluation = _load_json(metrics_dir, "evaluation_report.json")
    class_names = evaluation["class_names"]
    per_class = evaluation["per_class_metrics"]
    ordered = sorted(class_names, key=lambda c: per_class[c]["support"], reverse=True)
    precision = [per_class[c]["precision"] for c in ordered]
    recall = [per_class[c]["recall"] for c in ordered]
    f1 = [per_class[c]["f1-score"] for c in ordered]
    x = np.arange(len(ordered)); width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision", color=COLOR_TRAIN)
    ax.bar(x, recall, width, label="Recall", color=COLOR_VAL)
    ax.bar(x + width, f1, width, label="F1", color=COLOR_EMA)
    ax.set_xticks(x); ax.set_xticklabels(ordered, rotation=45, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall, and F1 (test set)")
    ax.legend(frameon=False); fig.tight_layout()
    _save(fig, plots_dir, "per_class_metrics.png")


def generate_run_plots(cfg, run_name: str) -> None:
    run_metrics = Path(cfg.output.metrics_dir) / run_name
    run_plots = Path(cfg.output.plots_dir) / run_name
    plot_loss_curve(run_metrics, run_plots)
    plot_accuracy_curve(run_metrics, run_plots)
    plot_learning_rate_schedule(run_metrics, run_plots)
    plot_confusion_matrix(run_metrics, run_plots)
    plot_per_class_metrics(run_metrics, run_plots)
