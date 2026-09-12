# Track 3 — Optimization toolkits: approach

Owner: _tbd_ | Branch: `track/optimization-toolkit` | Owns: `src/optimize/`, `src/export/`

## Order (do NOT start with Olive)
1. Baseline full pass: `python -m src.optimize.optimize --config configs/baseline_6class_mobilenetv3.yaml --run baseline --toolkit baseline`
   (export -> onnxsim? -> onnxruntime dynamic INT8 -> verify, acc drop must be <5pp).
2. Then ONE toolkit at a time: `--toolkit olive`, then `aimet`, then `modelopt`.
   Parent toolkit scripts live in `../tools/{olive,aimet,nvidia-modelopt}/` — wire them at the `quantize` step in `src/optimize/optimize.py`.
3. Every error stays in `optimization_log.jsonl` + `failures.md` with versions. Never force-push over a failed log.

## Success criteria
- [ ] Baseline INT8 pass green with `optimization_summary.json`
- [ ] At least one toolkit pass green, or a documented blocker with repro
- [ ] `failures.md` = error -> mitigation table leadership can read
