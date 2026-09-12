# Smoke test (CPU-friendly, ~minutes). Full data run is the same without --limit-per-class.
python scripts/run_baseline.py configs/baseline_6class_mobilenetv3.yaml --limit-per-class 200 --run smoke
