# Hand-Gesture Standard Pipeline (6-class)

Shared baseline for platform testing. Frozen contract: **6 classes**
(`dislike, fist, like, ok, palm, peace`, `data_version: v1-6class`),
**3 models** (`mobilenetv3_small_100`, `resnet18`, `mobilevit_xxs`).

## Quickstart

```bash
pip install -r requirements.txt
# smoke (200 imgs/class, CPU-friendly):
python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --limit-per-class 200 --run smoke
# full:
python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml
python scripts/run_baseline.py configs/baseline_6class_resnet18.yaml
python scripts/run_baseline.py configs/baseline_6class_mobilevit_xxs.yaml
pytest tests/ -q
```

## Stage order (one run)

`download_dataset` (18->6 remap + manifest) -> `preprocess`
(stratified split + report) -> `train` (single baseline, early stopping,
EMA-aware) -> `evaluate` (acc, per-class, ECE) -> `plots` -> `to_onnx`
(with torch/ONNX parity check).

## Team tracks (each owns one dir + one docs/ folder)

| Track | Branch | Owns | Entry point |
|---|---|---|---|
| HPO Hyperband/BOHB | `track/hpo-hyperband-bohb` | `src/hpo/` | `python -m src.hpo.run_hpo --config configs/baseline_6class_mobilenetv3.yaml --trials 27` |
| Offline profiling | `track/offline-profiling` | `src/profile/` | `python -m src.profile.offline_profile --config ... --run baseline` |
| Optimization toolkits | `track/optimization-toolkit` | `src/optimize/`, `src/export/` | `python -m src.optimize.optimize --config ... --run baseline` |

Shared code (`src/data`, `src/models`, `src/training`) changes require PR review.
Each track commits only `metrics/*.json`, `plots/*.png`, `manifest` files —
never `data/`, `*.pth`, `*.onnx` (gitignored).

## Review

Team leads review `integration/review-round-1` (merge of the three
`track/*` branches). Start at `REPORT.md`, then each
`docs/track-*/results.md`.
