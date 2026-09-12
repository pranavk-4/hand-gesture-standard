"""Full baseline pass: download -> preprocess -> train -> evaluate -> plots -> ONNX.

Replaces scripts/run_ablation_sequence.py (multi-stage study runner).
One config in, one scored + plotted + exported run out.

Usage:
    python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml
    python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --limit-per-class 200 --run smoke
    python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --skip-download --skip-preprocess
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="baseline_6class_mobilenetv3.yaml")
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(PROJECT_ROOT / "configs" / args.config)

    from src.data.download_dataset import download_and_extract
    from src.data.preprocess import run_preprocessing
    from src.training.train import run_training
    from src.evaluation.evaluate import run_evaluation
    from src.viz.plots import generate_run_plots
    from src.export.to_onnx import export_onnx

    if not args.skip_download:
        download_and_extract(cfg, limit_per_class=args.limit_per_class)
    if not args.skip_preprocess:
        run_preprocessing(cfg)

    summary = run_training(cfg, run_name=args.run)
    report = run_evaluation(Path(cfg.output.metrics_dir) / args.run)
    generate_run_plots(cfg, args.run)
    if not args.skip_onnx:
        try:
            export_onnx(cfg, args.run)
        except Exception:
            logger.exception("ONNX export failed — training artifacts above are still valid.")

    logger.info("DONE %s %s: test_acc=%.4f ece=%.4f",
                cfg.model.architecture, args.run, report["test_accuracy"],
                report["expected_calibration_error"])


if __name__ == "__main__":
    main()
