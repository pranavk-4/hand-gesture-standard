# Development Plan — Hand-Gesture Standard Pipeline

## Goal
One reproducible 6-class baseline that three teammates can extend in
parallel (HPO / profiling / optimization) and leadership can review in
one branch.

## Phase 0 — Freeze (this scaffold, Day 1)
- [x] 3 configs (`configs/baseline_6class_*.yaml`), `src/config.py` rejects non-6-class edits
- [x] 18->6 remap in `src/data/download_dataset.py` + `tests/test_class_map.py`
- [x] 3-arch factory (`mobilenetv3_small_100`, `resnet18`, `mobilevit_xxs`) + `tests/test_factory.py`
- [x] De-ablated `src/training/train.py` (`overrides` + `max_epochs_override` for Hyperband budgets)
- [x] `src/export/to_onnx.py` with parity check, `src/profile/`, `src/hpo/`, `src/optimize/` stubs
- [ ] Run `pytest tests/ -q` (must be green, no GPU needed)
- [ ] Run smoke: `run_baseline.py ... --limit-per-class 200 --run smoke`
- [ ] `git init`, commit, tag `v0.1-scaffold`, push `main`

Verify: tests green; smoke produces `training_summary.json`,
`evaluation_report.json`, 5 PNGs, `model.onnx`.

## Phase 1 — Baselines (each owner trains ONE model, Days 2-3)
- MobilenetV3 (primary): full data, expect >85% test acc on 6 classes
- ResNet18 + MobileViT-XXS: confirm convergence, record wall-clock + params
- Commit `metrics/*/training_summary.json`, `evaluation_report.json`, plots.
  Do NOT commit checkpoints/ONNX.

Verify: all three `evaluation_report.json` present; per-class F1 eyeballed
for collapsed classes (peace vs peace_inverted is gone — confusion should drop).

## Phase 2 — Parallel tracks (worktrees, Days 4-10)
- Track 1 HPO: `pip install optuna`, run `--trials 9` pilot then 27.
  Commit `outputs/*/metrics/hpo/{trials.csv,best_config.yaml,hpo_summary.json}`.
  Success: best trial beats baseline val acc; pruned trials logged, not deleted.
- Track 2 profiling: run `offline_profile` on all 3 baselines, then diff
  against platform profiler. Fill `platform_profiler_output` +
  `platform_vs_manual_delta` in `profile_report.json`.
- Track 3 optimization: `--toolkit baseline` full pass first
  (export->simplify->quantize->verify, acc drop <5pp), THEN wire
  olive/aimet/modelopt one at a time, logging every error to
  `optimization_log.jsonl`.

Verify per track: see `docs/track-*/approach.md` checklists.

## Phase 3 — Integration review (Day 11+)
- `git checkout -b integration/review-round-1 main`
- `git merge --no-ff track/hpo-hyperband-bohb track/offline-profiling track/optimization-toolkit`
- Fill `REPORT.md` (numbers table + links to per-track results).
- Leadership reviews the integration branch only.

## Non-goals
No new classes, no 4th architecture, no ablation stages, no committed
datasets/weights. Anything crossing track boundaries goes through a PR
on `main` first.
