"""Post-training optimization — owned by Track 3 (optimization toolkits).

Pipeline order (each step logged to optimization_log.jsonl, failures KEPT):
  1. export      torch -> ONNX (reuses src/export/to_onnx.py)
  2. simplify    onnxsim polish (optional, skips gracefully if missing)
  3. quantize    dynamic INT8 via onnxruntime (always-available fallback).
                 Toolkit passes (Olive / AIMET / nvidia-modelopt) plug in
                 HERE behind --toolkit {olive,aimet,modelopt} — see
                 TOOLKIT_NOTES below. Default --toolkit baseline uses only
                 onnxruntime so the full pass succeeds on a clean machine.
  4. verify      accuracy of FP32 ONNX vs INT8 ONNX on the test split
                 (tolerance-gated, not eyeballed).

Usage:
    python -m src.optimize.optimize --config configs/baseline_6class_mobilenetv3.yaml --run baseline
    python -m src.optimize.optimize --config configs/baseline_6class_mobilenetv3.yaml --run baseline --toolkit olive

TOOLKIT_NOTES (fill in as you test each toolkit manually):
  - olive:      Olive workflow JSON lives in tools/olive/ (parent ModelOPS_VL).
                Known failure: <paste error + versions>. Mitigation: <what worked>.
  - aimet:      AIMET quantsim scripts live in tools/aimet/.
  - modelopt:   ModelOpt scripts live in tools/nvidia-modelopt/.
"""

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _log(log_path: Path, step: str, status: str, detail: dict | None = None):
    entry = {"ts": time.time(), "step": step, "status": status, "detail": detail or {}}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("[%s] %s: %s", step, status, detail or "")


def _evaluate_onnx(onnx_path: Path, cfg) -> float:
    """Test accuracy of an ONNX file on the processed test split."""
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from torchvision import transforms

    from src.data.datasets import IMAGENET_MEAN, IMAGENET_STD

    tf = transforms.Compose([
        transforms.CenterCrop(cfg.preprocessing.image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_root = Path(cfg.output.metrics_dir).parents[2] / "data" / "processed" / "test"
    # Fallback: resolve via DATA_DIR (metrics_dir may be customized).
    if not test_root.exists():
        from src.config import DATA_DIR
        test_root = DATA_DIR / "processed" / "test"
    class_names = list(cfg.dataset.class_names)
    correct = total = 0
    for label, cname in enumerate(class_names):
        for img_path in sorted((test_root / cname).glob("*.jpg")):
            img = Image.open(img_path).convert("RGB")
            x = tf(img).unsqueeze(0).numpy()
            pred = int(np.argmax(sess.run(None, {"input": x})[0], axis=1)[0])
            correct += (pred == label)
            total += 1
    return correct / total if total else 0.0


def run_optimize(cfg, run_name: str, toolkit: str = "baseline") -> dict:
    from src.export.to_onnx import export_onnx

    ckpt_dir = Path(cfg.output.checkpoints_dir) / run_name
    log_path = ckpt_dir / "optimization_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    # 1. export
    try:
        export_onnx(cfg, run_name)
        _log(log_path, "export", "ok")
    except Exception as e:
        _log(log_path, "export", "failed", {"error": str(e)})
        raise

    onnx_path = ckpt_dir / "model.onnx"

    # 2. simplify (optional)
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(str(onnx_path))
        simp, ok = simplify(model)
        if ok:
            onnx.save(simp, str(onnx_path))
            _log(log_path, "simplify", "ok")
        else:
            _log(log_path, "simplify", "skipped", {"reason": "simplify returned ok=False"})
    except ImportError:
        _log(log_path, "simplify", "skipped", {"reason": "onnxsim not installed"})
    except Exception as e:
        _log(log_path, "simplify", "failed", {"error": str(e)})

    # 3. quantize
    int8_path = ckpt_dir / "model_int8.onnx"
    if toolkit == "baseline":
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
            _log(log_path, "quantize", "ok", {"toolkit": "onnxruntime-dynamic"})
        except Exception as e:
            _log(log_path, "quantize", "failed", {"error": str(e)})
            raise
    elif toolkit in ("olive", "aimet", "modelopt"):
        _log(log_path, "quantize", "todo",
              {"toolkit": toolkit, "hint": "Wire tools/ parent scripts here; log every error, do not delete."})
        raise NotImplementedError(
            f"Toolkit '{toolkit}' not wired yet. See TOOLKIT_NOTES in this file. "
            "Run --toolkit baseline for the guaranteed full pass first.")
    else:
        raise ValueError(f"Unknown toolkit {toolkit!r}")

    # 4. verify
    fp32_acc = _evaluate_onnx(onnx_path, cfg)
    int8_acc = _evaluate_onnx(int8_path, cfg) if int8_path.exists() else None
    drop = (fp32_acc - int8_acc) if int8_acc is not None else None
    _log(log_path, "verify", "ok",
          {"fp32_acc": fp32_acc, "int8_acc": int8_acc, "acc_drop": drop})

    summary = {
        "toolkit": toolkit, "run_name": run_name,
        "fp32_test_accuracy": fp32_acc, "int8_test_accuracy": int8_acc,
        "accuracy_drop": drop,
        "artifacts": [str(onnx_path), str(int8_path) if int8_path.exists() else None],
        "pass": drop is not None and drop < 0.05,
    }
    with open(ckpt_dir / "optimization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    from src.config import PROJECT_ROOT, load_config
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline_6class_mobilenetv3.yaml")
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--toolkit", default="baseline", choices=["baseline", "olive", "aimet", "modelopt"])
    args = ap.parse_args()
    cfg = load_config(PROJECT_ROOT / "configs" / args.config)
    print(json.dumps(run_optimize(cfg, args.run, toolkit=args.toolkit), indent=2))
