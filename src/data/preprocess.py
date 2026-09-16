"""Turns data/raw/<6 classes>/ into data/processed/{train,val,test}/.

Ported verbatim from hand_gesture_classifier/src/data/preprocess.py
(integrity check, EXIF fix, canonical resize, stratified split, JSON
report) — input shape is identical, only the class list is now 6.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class PreprocessReport:
    total_raw_images: int = 0
    dropped_corrupt: int = 0
    dropped_paths: list[str] = field(default_factory=list)
    class_counts_raw: dict[str, int] = field(default_factory=dict)
    class_counts_by_split: dict[str, dict[str, int]] = field(default_factory=dict)


def _is_valid_image(path, min_pixel_std: float) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img = img.convert("L")
            extrema = img.getextrema()
            if extrema[1] - extrema[0] < min_pixel_std:
                return False
        return True
    except Exception:
        return False


def _load_clean_and_resize(path, size: int) -> Image.Image:
    """Letterbox to a square `size` canvas: scale the WHOLE image so its long
    side is `size`, then pad the short side. HaGRID frames the hand anywhere
    in a room-scale photo and ships no bounding boxes, so the old
    resize-short-side + center-crop silently cropped hands out of frame
    (worst on palm/peace/ok). Padding preserves the whole subject with no
    aspect-ratio distortion — nothing gets clipped."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    w, h = image.size
    scale = size / max(w, h)
    resized = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    rw, rh = resized.size
    canvas.paste(resized, ((size - rw) // 2, (size - rh) // 2))
    return canvas


def run_preprocessing(cfg) -> PreprocessReport:
    from src.config import DATA_DIR
    raw_dir = DATA_DIR / "raw"
    processed_dir = DATA_DIR / "processed"

    class_names = list(cfg.dataset.class_names)
    canonical_size = cfg.preprocessing.image_size + 32
    report = PreprocessReport()

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    kept_by_class: dict[str, list] = {c: [] for c in class_names}
    for class_name in class_names:
        raw_paths = sorted((raw_dir / class_name).glob("*.jpg"))
        report.class_counts_raw[class_name] = len(raw_paths)
        report.total_raw_images += len(raw_paths)
        for path in raw_paths:
            if not _is_valid_image(path, cfg.preprocessing.min_pixel_std):
                report.dropped_corrupt += 1
                report.dropped_paths.append(str(path))
                continue
            kept_by_class[class_name].append(path)

    val_frac, test_frac, seed = cfg.dataset.val_fraction, cfg.dataset.test_fraction, cfg.dataset.random_seed
    for split_name in ("train", "val", "test"):
        for class_name in class_names:
            (processed_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

    for class_name, paths in kept_by_class.items():
        train_paths, holdout_paths = train_test_split(paths, test_size=val_frac + test_frac, random_state=seed)
        val_paths, test_paths = train_test_split(
            holdout_paths, test_size=test_frac / (val_frac + test_frac), random_state=seed
        )
        for split_name, split_paths in (("train", train_paths), ("val", val_paths), ("test", test_paths)):
            out_dir = processed_dir / split_name / class_name
            for i, src_path in enumerate(split_paths):
                _load_clean_and_resize(src_path, canonical_size).save(
                    out_dir / f"{class_name}_{i:05d}.jpg", "JPEG", quality=95
                )
            report.class_counts_by_split.setdefault(split_name, {})[class_name] = len(split_paths)

    report_path = cfg.output.metrics_dir / "preprocessing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=2)
    logger.info("Preprocessing complete: %d raw, %d dropped.", report.total_raw_images, report.dropped_corrupt)
    return report


if __name__ == "__main__":
    import sys
    from src.config import load_config, PROJECT_ROOT
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline_6class_mobilenetv3.yaml"
    cfg = load_config(PROJECT_ROOT / "configs" / name)
    run_preprocessing(cfg)
