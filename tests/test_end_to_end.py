"""End-to-end plumbing test on the tiny synthetic dataset (see
scripts/make_tiny_dataset.py). Runs train(1 epoch) -> eval -> plots ->
onnx -> profile -> optimize-baseline on CPU with random init.

Slow (builds mobilenetv3 + trains) — marked for explicit runs, not the
default unit suite:
    pytest tests/test_end_to_end.py -q
"""

import json
from pathlib import Path

from src.config import PROJECT_ROOT, load_config


def test_end_to_end_plumbing(tmp_path=None):
    import torch
    from scripts.make_tiny_dataset import build_tiny_dataset
    from src.data.datasets import build_dataloaders
    from src.training.train import run_training
    from src.evaluation.evaluate import run_evaluation
    from src.viz.plots import generate_run_plots
    from src.export.to_onnx import export_onnx
    from src.profile.offline_profile import offline_profile
    from src.optimize.optimize import run_optimize

    cfg = load_config(PROJECT_ROOT / "configs" / "baseline_6class_mobilenetv3.yaml")
    cfg.model.pretrained = False
    cfg.training.max_epochs = 1
    cfg.training.batch_size = 4
    cfg.training.num_workers = 0
    cfg.training.use_amp = False
    cfg.training.ema = False
    cfg.training.early_stopping_patience = 5

    build_tiny_dataset()  # self-contained: no HF download, no fixtures

    run_name = "e2e_verify"
    summary = run_training(cfg, run_name=run_name)
    assert summary["epochs_trained"] == 1
    assert len(summary["test_predictions"]) > 0

    report = run_evaluation(Path(cfg.output.metrics_dir) / run_name)
    assert report["test_accuracy"] >= 0.0

    generate_run_plots(cfg, run_name)
    for png in ("loss_curve.png", "accuracy_curve.png", "confusion_matrix.png"):
        assert (Path(cfg.output.plots_dir) / run_name / png).exists(), png

    export_onnx(cfg, run_name)
    assert (Path(cfg.output.checkpoints_dir) / run_name / "model.onnx").exists()

    prof = offline_profile(cfg, run_name, repeats=5, warmup=2)
    assert prof["param_count"] > 0

    opt = run_optimize(cfg, run_name, toolkit="baseline")
    assert opt["fp32_test_accuracy"] >= 0.0
