# Track 1 — HPO (Hyperband / BOHB): approach

Owner: _tbd_ | Branch: `track/hpo-hyperband-bohb` | Owns: `src/hpo/`

## What to test
1. Hyperband pruner aggression (does `reduction_factor: 3` kill late-bloomers? compare rung survivors vs full-budget rerun).
2. BOHB ~= TPE sampler + Hyperband pruner: does TPE beat random search at 27 trials? (ablate sampler).
3. Failure modes: OOM at high LR, NaN loss, pruner starving one region of space.

## Steps
1. `pip install optuna>=3.0`
2. Pilot: `python -m src.hpo.run_hpo --config configs/baseline_6class_mobilenetv3.yaml --trials 9`
3. Full: `--trials 27`. Never edit the baseline config per trial — overrides flow through `run_training(overrides=...)`.
4. Commit `outputs/*/metrics/hpo/{trials.csv,best_config.yaml,hpo_summary.json}`.

## Success criteria
- [ ] `trials.csv` has ALL trials incl. PRUNED/FAIL (no filtering)
- [ ] Best trial beats frozen baseline val acc
- [ ] `failures.md` lists every error + mitigation
