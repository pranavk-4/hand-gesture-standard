"""Offline model profiling — owned by Track 2 (profiling teammate).

Contract:
  Input:  trained checkpoint (<checkpoints_dir>/<run>/best_model.pth)
          or exported ONNX (model.onnx)
  Output: outputs/<arch>/metrics/<run>/profile_report.json
          {param_count, flops_estimate, cpu_latency_ms_p50/p90,
           peak_ram_mb, onnx_size_mb, device, notes}

Manual-testing friendly: every number records how it was measured
(device string, batch size, warmup/repeat counts) so the platform's
automated profiler can later be diffed against this hand-rolled report.

Usage:
    python -m src.profile.offline_profile --config configs/baseline_6class_mobilenetv3.yaml --run baseline
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _try_flops(model, image_size: int):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, image_size, image_size)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs * 2)  # MACs -> FLOPs
    except Exception as e:
        logger.warning("FLOPs estimate unavailable (%s). Install thop for FLOPs.", e)
        return None


def offline_profile(cfg, run_name: str, repeats: int = 100, warmup: int = 20) -> dict:
    from src.models.factory import build_model

    ckpt_path = Path(cfg.output.checkpoints_dir) / run_name / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(cfg.model.architecture, num_classes=len(cfg.dataset.class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image_size = cfg.preprocessing.image_size
    dummy = torch.randn(1, 3, image_size, image_size)

    param_count = sum(p.numel() for p in model.parameters())
    flops = _try_flops(model, image_size)

    onnx_path = ckpt_path.parent / "model.onnx"
    onnx_size_mb = onnx_path.stat().st_size / 1e6 if onnx_path.exists() else None

    # CPU latency: warmup then timed repeats, batch=1.
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        lat = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(dummy)
            lat.append((time.perf_counter() - t0) * 1000)
    lat_sorted = sorted(lat)
    report = {
        "architecture": cfg.model.architecture,
        "run_name": run_name,
        "device": "cpu",
        "batch_size": 1,
        "warmup_iters": warmup,
        "repeat_iters": repeats,
        "param_count": param_count,
        "flops_estimate": flops,
        "flops_tool": "thop" if flops is not None else None,
        "cpu_latency_ms_p50": lat_sorted[len(lat_sorted) // 2],
        "cpu_latency_ms_p90": lat_sorted[int(len(lat_sorted) * 0.9)],
        "onnx_size_mb": onnx_size_mb,
        "source_checkpoint": str(ckpt_path),
        "notes": "Hand-rolled baseline. Platform profiler output must be diffed against this file.",
        # Track 2 fills these in as they integrate the platform profiler:
        "platform_profiler_output": None,
        "platform_vs_manual_delta": None,
        "failure_patterns": [],
    }
    out = Path(cfg.output.metrics_dir) / run_name / "profile_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Profile: params=%d flops=%s p50=%.2fms p90=%.2fms -> %s",
                param_count, flops, report["cpu_latency_ms_p50"], report["cpu_latency_ms_p90"], out)
    return report


if __name__ == "__main__":
    from src.config import PROJECT_ROOT, load_config
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline_6class_mobilenetv3.yaml")
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--repeats", type=int, default=100)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / "configs" / cfg_path.name
    cfg = load_config(cfg_path)
    offline_profile(cfg, args.run, repeats=args.repeats)
