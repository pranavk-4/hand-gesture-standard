"""Full baseline pass: download -> preprocess -> train -> evaluate -> plots -> ONNX.

Replaces scripts/run_ablation_sequence.py (multi-stage study runner).
One config in, one scored + plotted + exported run out.

Usage:
    python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml
    python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --limit-per-class 200 --run smoke
    python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --skip-download --skip-preprocess
    # Fast offline check (no download, 2 epochs, random init):
    python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --skip-download --skip-preprocess --max-epochs 2 --no-pretrained --run verify
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
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Override cfg.training.max_epochs (e.g. 2 for a fast verify pass).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Random init instead of ImageNet weights (offline / CI environments).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / "configs" / cfg_path.name
    cfg = load_config(cfg_path)
    if args.max_epochs is not None:
        cfg.training.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.no_pretrained:
        cfg.model.pretrained = False

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
