"""Downloads HaGRID-sample-30k and extracts ONLY the standard 6 classes.

Fixes the 18->6 trap in the research pipeline
(hand_gesture_classifier/src/data/download_dataset.py used the raw
parquet `label` int directly as an index into `class_names`, which is
only correct for the full 18-class list). Here the parquet label is
first mapped through FULL_18CLASS_NAMES, then filtered + remapped to
the frozen STANDARD_CLASSES index.
"""

import io
import logging

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

logger = logging.getLogger(__name__)

FULL_18CLASS_NAMES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop",
    "stop_inverted", "three", "three2", "two_up", "two_up_inverted",
]


def _decode_image(cell) -> Image.Image:
    if isinstance(cell, dict):
        raw_bytes = cell.get("bytes")
        if raw_bytes is not None:
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        path = cell.get("path")
        if path:
            return Image.open(path).convert("RGB")
    raise ValueError(f"Unrecognized image cell format: {type(cell)}")


def build_class_index_map(keep: list[str]) -> dict[int, int]:
    """Map source parquet label int -> new 6-class index. Raises on dupes."""
    mapping = {}
    for src_idx, name in enumerate(FULL_18CLASS_NAMES):
        if name in keep:
            mapping[src_idx] = keep.index(name)
    missing = [c for c in keep if c not in FULL_18CLASS_NAMES]
    if missing:
        raise ValueError(f"Keep-classes not in source 18: {missing}")
    if len(set(keep)) != len(keep):
        raise ValueError(f"Duplicate class names in keep list: {keep}")
    return mapping


def download_and_extract(cfg, limit_per_class: int | None = None) -> int:
    from src.config import DATA_DIR
    raw_dir = DATA_DIR / "raw"

    keep = list(cfg.dataset.class_names)
    index_map = build_class_index_map(keep)
    logger.info("Class remap (src->new): %s", index_map)

    revision = getattr(cfg.dataset, "hf_revision", None)
    filenames = cfg.dataset.hf_filename
    if isinstance(filenames, str):
        filenames = [filenames]
    parquet_paths = [
        hf_hub_download(
            repo_id=cfg.dataset.hf_repo_id, repo_type="dataset",
            filename=fn, revision=revision,
        )
        for fn in filenames
    ]
    df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)

    raw_dir.mkdir(parents=True, exist_ok=True)
    for class_name in keep:
        (raw_dir / class_name).mkdir(exist_ok=True)

    counters = {c: 0 for c in keep}
    skipped = 0
    for _, row in df.iterrows():
        src_label = int(row["label"])
        if src_label not in index_map:
            skipped += 1
            continue
        class_name = keep[index_map[src_label]]
        if limit_per_class is not None and counters[class_name] >= limit_per_class:
            continue
        image = _decode_image(row["image"])
        out_path = raw_dir / class_name / f"img_{counters[class_name]:05d}.jpg"
        image.save(out_path, "JPEG", quality=95)
        counters[class_name] += 1

    total = sum(counters.values())
    logger.info("Wrote %d images across %d classes to %s (skipped %d non-keep)",
                total, len(keep), raw_dir, skipped)
    manifest = {
        "data_version": getattr(cfg.dataset, "data_version", "unknown"),
        "keep_classes": keep,
        "index_map": {str(k): v for k, v in index_map.items()},
        "counts": counters,
        "skipped_non_keep": skipped,
    }
    import json
    with open(raw_dir / "download_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return total


if __name__ == "__main__":
    import sys
    from src.config import load_config, PROJECT_ROOT
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline_6class_mobilenetv3.yaml"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cfg = load_config(PROJECT_ROOT / "configs" / name)
    download_and_extract(cfg, limit_per_class=limit)
