# Hand-Gesture Standard Pipeline (6-class)

Shared baseline for platform testing. Frozen contract: **6 classes**
(`dislike, fist, like, ok, palm, peace`, `data_version: v1-6class`),
**3 models** (`mobilenetv3_small_100`, `resnet18`, `mobilevit_xxs`).
Verified end-to-end on CPU: train -> eval -> plots -> ONNX (torch/ORT
parity-checked) -> profile -> INT8 quantize.

## First hour (everyone)

```bash
git clone <repo-url> hand-gesture-standard && cd hand-gesture-standard
pip install -r requirements.txt
pytest tests/ -q                                   # must be green
# Offline plumbing check (no download, ~2 min):
python scripts/make_tiny_dataset.py
python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --skip-download --skip-preprocess --max-epochs 1 --batch-size 4 --no-pretrained --run verify
```

Real data (needs HF access):

```bash
# Smoke (~200 imgs/class):
python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --limit-per-class 200 --run smoke
# Full baselines (one per model):
python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml
python scripts/run_baseline.py configs/baseline_6class_resnet18.yaml
python scripts/run_baseline.py configs/baseline_6class_mobilevit_xxs.yaml
```

One config in, one scored + plotted + exported run out:
`download_dataset` (18->6 remap + manifest) -> `preprocess` (stratified
split + report) -> `train` (early stopping, EMA-aware) -> `evaluate`
(acc, per-class, ECE) -> `plots` -> `to_onnx` (parity check).

## Workstreams (one branch + one folder each)

| You | Branch | Your folder + entry point | Done when |
|---|---|---|---|
| HPO (Hyperband/BOHB) | `track/hpo-hyperband-bohb` | `src/hpo/` → `python -m src.hpo.run_hpo --config configs/baseline_6class_mobilenetv3.yaml --trials 9` (needs `pip install optuna`) | `outputs/*/metrics/hpo/{trials.csv,best_config.yaml,hpo_summary.json}` committed, best beats baseline, pruned trials kept |
| Profiling | `track/offline-profiling` | `src/profile/` → `python -m src.profile.offline_profile --config <each> --run baseline` | 3x `profile_report.json` + platform-vs-manual delta |
| Optimization | `track/optimization-toolkit` | `src/optimize/` → `python -m src.optimize.optimize --config ... --run baseline --toolkit baseline` first, then `olive` / `aimet` / `modelopt` | `optimization_summary.json` with INT8 drop <5pp, every error in `optimization_log.jsonl` |

Approaches and failure logs live in our separate internal task file —
the repo holds code, results, and metrics. Per-track results summaries:
`docs/track-{hpo,profiling,optimization}/results.md`. Leadership review
starts at `REPORT.md` on the integration branch.

## Workflow

Worktrees (no stashing) — from the repo root:

```powershell
# Windows
.\scripts\setup_worktrees.ps1
# Linux / macOS
bash scripts/setup_worktrees.sh
```

This creates `../wt-hpo`, `../wt-profiling`, `../wt-optimization` on
their branches.

## Ground rules

- `main` is protected. Only PRs from `track/*`.
- Shared code (`src/data/`, `src/models/`, `src/training/`,
  `src/config.py`, `configs/`) changes require PR review. Each track
  commits freely inside its own folder + results doc.
- 6-class set and `data_version` are frozen — `src/config.py` rejects edits.
- Commit metrics/plots/manifests/summaries only. Never `data/`,
  `*.pth`, `*.onnx`, `*.pt` (gitignored).
- Every failure is a log row, not a deleted file — record it in the
  internal task file with versions and repro.

## Integration (maintainer)

```bash
git checkout -b integration/review-round-1 main
git merge --no-ff track/hpo-hyperband-bohb track/offline-profiling track/optimization-toolkit
# fill REPORT.md numbers table, open PR, request leadership review
```
