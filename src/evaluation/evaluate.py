"""Scored report from a run's test predictions: confusion matrix,
per-class precision/recall/F1, ECE. Verbatim port of
hand_gesture_classifier/src/evaluation/evaluate.py (class-count agnostic).
"""

import json
import logging

from sklearn.metrics import classification_report, confusion_matrix

from src.training.metrics import expected_calibration_error

logger = logging.getLogger(__name__)


def build_evaluation_report(summary: dict) -> dict:
    preds, labels = summary["test_predictions"], summary["test_labels"]
    class_names = summary["class_names"]
    confidences = summary["test_confidences"]
    correct = [p == l for p, l in zip(preds, labels)]

    return {
        "run_name": summary["run_name"], "architecture": summary["architecture"],
        "test_accuracy": summary["test_accuracy"],
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "class_names": class_names,
        "per_class_metrics": classification_report(labels, preds, target_names=class_names, output_dict=True),
        "expected_calibration_error": expected_calibration_error(confidences, correct),
        "epochs_trained": summary["epochs_trained"],
        "best_epoch": summary["best_epoch"],
        "best_val_accuracy": summary["best_val_accuracy"],
        "total_wall_clock_seconds": summary["total_wall_clock_seconds"],
        "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
        "param_count": summary["param_count"],
    }


def run_evaluation(metrics_dir) -> dict:
    with open(metrics_dir / "training_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    report = build_evaluation_report(summary)
    out_path = metrics_dir / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("%s %s: test_accuracy=%.4f ece=%.4f", summary["architecture"], summary["run_name"],
                report["test_accuracy"], report["expected_calibration_error"])
    return report
