# Track 3 — failure log (the main deliverable alongside the passing run)

| Date | Step (export/simplify/quantize/verify) | Toolkit + versions | Error (paste) | Mitigation tried | Outcome |
|---|---|---|---|---|---|
| 2026-09-12 | export (re-export over existing `model.onnx`) | torch 2.x dynamo exporter + onnxruntime, Windows cp1252 console | `OSError: [Errno 22]` on `model.onnx.data` — prior ORT session held the file mapped | `del sess + gc.collect()` after parity check; delete stale `model.onnx(.data)` before export (`src/export/to_onnx.py`) | Fixed, covered by `tests/test_end_to_end.py` |
| 2026-09-12 | export (first run, Windows) | torch.onnx dynamo exporter + wandb console capture | `UnicodeEncodeError: 'charmap' codec can't encode '✅'` — exporter status glyph vs cp1252 console | Swallow exporter stdout/stderr via `redirect_stdout(StringIO())` in `export_onnx` | Fixed, export parity 3.7e-08 |
