"""Builds a tiny synthetic processed dataset so the full pipeline can be
verified WITHOUT downloading HaGRID (offline / CI / pre-kickoff check).

Writes data/processed/{train,val,test}/<6 classes>/*.jpg with random-noise
images. Labels are directory names, so ImageFolder ordering matches
STANDARD_CLASSES. NOT for accuracy — only for plumbing:
train -> eval -> plots -> onnx -> profile -> optimize.

Usage:
    python scripts/make_tiny_dataset.py --train-per-class 12 --val-per-class 4 --test-per-class 4
"""

import argparse
import random
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, STANDARD_CLASSES


def build_tiny_dataset(train_per_class: int = 12, val_per_class: int = 4,
                       test_per_class: int = 4, size: int = 256, seed: int = 0) -> Path:
    """Generate random-noise JPEGs into data/processed/. Returns processed dir."""
    rng = random.Random(seed)
    splits = {"train": train_per_class, "val": val_per_class, "test": test_per_class}
    for split, per_class in splits.items():
        for cls in STANDARD_CLASSES:
            d = DATA_DIR / "processed" / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*.jpg"):
                f.unlink()
            for i in range(per_class):
                px = bytes(rng.randrange(256) for _ in range(size * size * 3))
                img = Image.frombytes("RGB", (size, size), px)
                img.save(d / f"{cls}_{i:05d}.jpg", "JPEG", quality=90)
    total = sum(splits.values()) * len(STANDARD_CLASSES)
    print(f"Tiny dataset ready: {total} images "
          f"({splits['train']}/{splits['val']}/{splits['test']} per class) in {DATA_DIR / 'processed'}")
    return DATA_DIR / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=12)
    ap.add_argument("--val-per-class", type=int, default=4)
    ap.add_argument("--test-per-class", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build_tiny_dataset(args.train_per_class, args.val_per_class,
                       args.test_per_class, args.size, args.seed)


if __name__ == "__main__":
    main()
