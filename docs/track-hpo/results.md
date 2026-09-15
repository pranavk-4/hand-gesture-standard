# Track 1 — results

## Best config vs baseline
Smoke data 1200 imgs pretrained 15 epochs baseline: val 0.4056 test 0.3889.
Best is BOHB high 12-param 6 trials: val 0.6 test 0.5889 trial 4.
Beats baseline by 0.194 val and 0.2 test.
See `outputs/mobilenetv3_small/metrics/hpo/hpo_summary.json`.

## Pruner analysis
Baseline no HPO 15 epochs: val 0.4056 test 0.3889.
Hyperband low 4-param: 5 complete 1 pruned best val 0.5166 test 0.4889.
Hyperband high 12-param: 4 complete 2 pruned best val 0.3222 test 0.2944.
BOHB low 4-param: 5 complete 1 pruned best val 0.55 test 0.5055.
BOHB high 12-param: 5 complete 1 pruned best val 0.6 test 0.5889.
High space only pays with TPE. Random in 12D loses to 4D on 6 trials.
Pruner fires on repeated rung with spread. Single final report limits mid training kills.

## Platform notes (what the automated HPO must handle)
Sampler swap is one line: tpe for BOHB, random for Hyperband, same HyperbandPruner.
Max tunable without shared change is 12 keys: 4 continuous plus 8 bool flags.
trials.csv now logs test_accuracy test_loss epochs_trained budget for every trial including pruned.
Need per epoch reporting in train loop for true successive halving.
ONNX parity failed on smoke baseline with 0.011 error. Training artifacts remain valid.
