"""Loads a per-model baseline config and exposes it with attribute access.

Standard-pipeline version of hand_gesture_classifier/src/config.py:
- validates the frozen 6-class contract (STANDARD_CLASSES),
- validates data_version pin so HPO/profiling/optimization tracks
  can prove they trained on the same data.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Frozen standard set (alphabetical — must match ImageFolder sort order).
STANDARD_CLASSES = ["dislike", "fist", "like", "ok", "palm", "peace"]
DATA_VERSION = "v1-6class"


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def load_config(path: Path) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _to_namespace(raw)

    got = list(cfg.dataset.class_names)
    if got != STANDARD_CLASSES:
        raise ValueError(
            f"Config {path} class_names {got} != STANDARD_CLASSES {STANDARD_CLASSES}. "
            "The 6-class set is frozen; do not edit per-track."
        )
    if getattr(cfg.dataset, "data_version", None) != DATA_VERSION:
        raise ValueError(
            f"Config {path} data_version {getattr(cfg.dataset, 'data_version', None)!r} "
            f"!= {DATA_VERSION!r}."
        )

    cfg.output.checkpoints_dir = PROJECT_ROOT / cfg.output.checkpoints_dir
    cfg.output.metrics_dir = PROJECT_ROOT / cfg.output.metrics_dir
    cfg.output.plots_dir = PROJECT_ROOT / cfg.output.plots_dir
    for directory in (cfg.output.checkpoints_dir, cfg.output.metrics_dir, cfg.output.plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return cfg
