"""PyTorch checkpoint -> ONNX. Shared contract for profiling + optimization tracks.

Usage:
    python -m src.export.to_onnx --config configs/baseline_6class_mobilenetv3.yaml --run baseline

Reads <checkpoints_dir>/<run>/best_model.pth, writes
<checkpoints_dir>/<run>/model.onnx + export_report.json
(opset, input shape, class names, SHA of source checkpoint).

Verifies the ONNX output matches torch output within tolerance on one
dummy batch before declaring success — this catches the silent
opset/simplify breakages the optimization track is hunting.
"""

import argparse
import hashlib
import io
import json
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(cfg, run_name: str, opset: int = 17) -> dict:
    from src.models.factory import build_model

    ckpt_path = Path(cfg.output.checkpoints_dir) / run_name / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first: scripts/run_baseline.py")

    onnx_path = ckpt_path.parent / "model.onnx"
    architecture = cfg.model.architecture
    num_classes = len(cfg.dataset.class_names)
    image_size = cfg.preprocessing.image_size

    # A previous export's onnxruntime session can keep model.onnx.data
    # memory-mapped on Windows, making overwrite fail with Errno 22.
    # Remove stale outputs first so failures are loud here, not cryptic
    # deep inside the exporter.
    for stale in (onnx_path, onnx_path.with_suffix(".onnx.data")):
        try:
            if stale.exists():
                stale.unlink()
        except OSError:
            logger.exception("Cannot remove stale %s — is another process/session holding it?", stale)
            raise

    model = build_model(architecture, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    # torch.onnx prints unicode status glyphs (e.g. ✅) that crash Windows
    # cp1252 consoles / wandb console capture with UnicodeEncodeError — a
    # console bug, not a model bug. Swallow exporter chatter; real errors
    # still raise through.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
        )

    # Verify parity torch vs onnxruntime.
    try:
        import gc

        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": dummy.numpy()})[0]
        max_err = float(np.abs(torch_out - ort_out).max())
        # Release the session's handle on model.onnx(.data) promptly:
        # on Windows an open handle blocks the next export's overwrite.
        del sess
        gc.collect()
    except ImportError:
        max_err = None
        logger.warning("onnxruntime not installed — skipping parity check.")

    report = {
        "architecture": architecture,
        "run_name": run_name,
        "onnx_path": str(onnx_path),
        "opset": opset,
        "input_shape": [1, 3, image_size, image_size],
        "class_names": list(cfg.dataset.class_names),
        "source_checkpoint_sha256": _sha256(ckpt_path),
        "torch_onnx_max_abs_error": max_err,
    }
    with open(ckpt_path.parent / "export_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Exported %s (max_abs_error=%s)", onnx_path, max_err)
    if max_err is not None and max_err > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max_abs_error={max_err} > 1e-4")
    return report


if __name__ == "__main__":
    from src.config import PROJECT_ROOT, load_config
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline_6class_mobilenetv3.yaml")
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / "configs" / cfg_path.name
    cfg = load_config(cfg_path)
    export_onnx(cfg, args.run, opset=args.opset)
