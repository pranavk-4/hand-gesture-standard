# Team Brief — Hand-Gesture Standard Pipeline (share this page)

## What it is
One frozen 6-class baseline (`dislike, fist, like, ok, palm, peace`,
`data_version: v1-6class`) x 3 small models
(`mobilenetv3_small_100`, `resnet18`, `mobilevit_xxs`) that we all extend
in parallel to test platform features. Verified end-to-end on CPU on
2026-09-12: train -> eval -> plots -> ONNX (parity 3.7e-08) -> profile ->
INT8 quantize (pass=true). Random-noise data, so 16.7% acc is expected —
it proves plumbing, not quality.

## Your first hour (everyone)
```bash
git clone <repo-url> hand-gesture-standard && cd hand-gesture-standard
pip install -r requirements.txt
pytest tests/ -q                                   # fast unit suite, must be green
python scripts/make_tiny_dataset.py                # offline check data (no download)
python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --skip-download --skip-preprocess --max-epochs 1 --batch-size 4 --no-pretrained --run verify
```
Then the real data (needs HF access, ~30k images):
```bash
python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --limit-per-class 200 --run smoke
python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --run baseline   # full
```

## Workstreams (one branch + one folder each — do not cross-edit)

| You | Branch | Your folder + entry point | Done when |
|---|---|---|---|
| HPO (Hyperband/BOHB) | `track/hpo-hyperband-bohb` | `src/hpo/` → `python -m src.hpo.run_hpo --config configs/baseline_6class_mobilenetv3.yaml --trials 9` (needs `pip install optuna`) | `outputs/*/metrics/hpo/{trials.csv,best_config.yaml,hpo_summary.json}` committed, best beats baseline, pruned trials kept |
| Profiling | `track/offline-profiling` | `src/profile/` → `python -m src.profile.offline_profile --config <each> --run baseline` | 3x `profile_report.json` + platform-vs-manual delta |
| Optimization | `track/optimization-toolkit` | `src/optimize/` → `python -m src.optimize.optimize --config ... --run baseline --toolkit baseline` first, then `olive`/`aimet`/`modelopt` | `optimization_summary.json` with INT8 drop <5pp, every error in `optimization_log.jsonl` |

Setup: `scripts/setup_worktrees.ps1` (Windows) or `.sh` (Linux) creates
one worktree per branch. See `CONTRIBUTING.md` for the merge + review flow.
Docs per track: `docs/track-{hpo,profiling,optimization}/{approach,failures,results}.md`.
Leadership reviews `integration/review-round-1` starting at `REPORT.md`.

## Ground rules
- `main` is protected. Shared code (`src/data`, `src/models`, `src/training`) = PR review.
- 6-class set and `data_version` are frozen — `src/config.py` rejects edits.
- Commit metrics/plots/manifests only. Never `data/`, `*.pth`, `*.onnx` (gitignored).
- Every failure is a log row, not a deleted file. Failure tables ARE deliverables.
