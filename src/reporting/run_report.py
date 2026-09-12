"""Renders one run's evaluation_report.json + training_history.json into a
Markdown section. Ported from hand_gesture_classifier/src/reporting/run_report.py,
`stage` field dropped (no ablation stages in the standard pipeline).
"""


def render_run_section(evaluation_report: dict, history: list[dict]) -> str:
    r = evaluation_report
    best = max(history, key=lambda h: h["val_accuracy"])
    lines = [
        f"### `{r['run_name']}` ({r['architecture']})",
        "",
        f"Test accuracy: **{r['test_accuracy']:.4f}** | "
        f"ECE: **{r['expected_calibration_error']:.4f}** | "
        f"Best epoch: {r['best_epoch']}/{r['epochs_trained']} | "
        f"Wall clock: {r['total_wall_clock_seconds']:.0f}s",
        "",
        f"Best-epoch generalization gap: {best['generalization_gap']:.4f} | "
        f"Grad norm at best epoch (head/backbone): "
        f"{best['grad_norm_head']:.3f} / {best['grad_norm_backbone']:.3f}",
        "",
        "| Class | Precision | Recall | F1 |",
        "|---|---|---|---|",
    ]
    for class_name in r["class_names"]:
        m = r["per_class_metrics"][class_name]
        lines.append(f"| {class_name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} |")

    return "\n".join(lines) + "\n"
