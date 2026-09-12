# Track 2 — Offline profiling: approach

Owner: _tbd_ | Branch: `track/offline-profiling` | Owns: `src/profile/`

## What to test
1. Hand-rolled numbers first (`offline_profile.py`): params, FLOPs (thop, optional), CPU p50/p90 latency, ONNX size.
2. Then run the platform's profiler on the SAME checkpoint+ONNX and diff into `platform_vs_manual_delta`.
3. Failure modes: missing ops in ONNX, dynamic-axes slowdown, thop unsupported layers, device mismatch (CPU here vs target edge).

## Steps
1. `python -m src.export.to_onnx --config configs/baseline_6class_mobilenetv3.yaml --run baseline` (repeat per model)
2. `python -m src.profile.offline_profile --config <each> --run baseline`
3. Fill `platform_profiler_output` in each `profile_report.json` after the platform run.
4. Commit `profile_report.json` files only.

## Success criteria
- [ ] 3 `profile_report.json` (one per model) with measurement metadata
- [ ] Platform-vs-manual delta documented
- [ ] `failures.md` lists every profiler error + workaround
