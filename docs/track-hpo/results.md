# Track 1 — results

## Best config vs baseline
Proper data 4200 train 900 val 900 test letterbox pretrained baseline 60 epochs: val 0.9022 test 0.8678.
Best so far is Hyperband low trial 000: val 0.9056 test 0.8811, ahead of baseline by 0.003 val and 0.013 test.
Old smoke numbers are superseded and kept below for history.
See `outputs/mobilenetv3_small/metrics/hpo/hyperband_low/` per trial summaries.

## Pruner analysis
Proper baseline no HPO 60 epochs: val 0.9022 test 0.8678, 1843s.
Hyperband low 4-param partial 4 of 6 trials: trial_000 val 0.9056 test 0.8811 30ep, trial_001 val 0.7256 test 0.72 30ep, trial_002 val 0.74 test 0.7178 30ep, trial_003 val 0.8933 test 0.8756 25ep early stopped. Trial_004 cut by tool timeout, study csv not yet written.
Hyperband high, BOHB low, BOHB high on proper data: not yet run. Each full 30 epoch trial costs about 15 min on this GPU, so 6 trials per config is about 90 min.
History on smoke 1200 imgs 15 epochs: baseline val 0.4056 test 0.3889. Hyperband low best val 0.5166 test 0.4889. Hyperband high best val 0.3222 test 0.2944. BOHB low best val 0.55 test 0.5055. BOHB high best val 0.6 test 0.5889. High space only paid with TPE.

## Platform notes (what the automated HPO must handle)
Sampler swap is one line: tpe for BOHB, random for Hyperband, same HyperbandPruner.
Max tunable without shared change is 12 keys: 4 continuous plus 8 bool flags.
trials.csv now logs test_accuracy test_loss epochs_trained budget for every trial including pruned.
Per epoch pruning is live via epoch_callback in train.py with trial.report each epoch.
ONNX parity now passes with predicted class agreement on proper baseline.
Each tagged HPO run writes its own subdir under metrics hpo, so runs do not overwrite.
