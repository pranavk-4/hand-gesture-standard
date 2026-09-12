# Contributing — branch, worktree, and review flow

## Branches
- `main` — protected. Only PRs from `track/*`. Must keep `pytest tests/ -q` green.
- `track/hpo-hyperband-bohb`, `track/offline-profiling`, `track/optimization-toolkit` — one owner each.
- `integration/review-round-1` — maintainer-only merge of the three tracks for leadership review.

## Worktrees (parallel checkouts, no stashing)
```powershell
# Windows
.\scripts\setup_worktrees.ps1
# Linux / macOS
bash scripts/setup_worktrees.sh
```
This creates `../wt-hpo`, `../wt-profiling`, `../wt-optimization` on their branches.

## Ownership
- Shared (PR review required): `src/data/`, `src/models/`, `src/training/`, `src/config.py`, `configs/`.
- Track-owned (owner commits freely): Track 1 `src/hpo/` + `docs/track-hpo/`;
  Track 2 `src/profile/` + `docs/track-profiling/`; Track 3 `src/optimize/` +
  `src/export/` + `docs/track-optimization/`.

## What to commit
DO commit: `outputs/*/metrics/*.json`, `outputs/*/plots/*.png`,
`outputs/*/metrics/hpo/*.{csv,yaml,json}`, `*_report.json`, `*_summary.json`,
`optimization_log.jsonl`, docs.
NEVER commit: `data/`, `*.pth`, `*.onnx`, `*.pt`, `*.db*` (see `.gitignore`).

## Definition of done per track
- HPO: `trials.csv` contains every trial incl. PRUNED/FAIL; `best_config.yaml` reproduces the winner via `run_training(overrides=...)`.
- Profiling: 3 `profile_report.json` with device/batch/warmup metadata + `platform_vs_manual_delta` filled or explicitly TBD with reason.
- Optimization: baseline `--toolkit baseline` green first; toolkit attempts append to `optimization_log.jsonl` — failures kept with versions.

## Integration (maintainer)
```bash
git checkout -b integration/review-round-1 main
git merge --no-ff track/hpo-hyperband-bohb track/offline-profiling track/optimization-toolkit
# fill REPORT.md numbers table, open PR, request leadership review
```
