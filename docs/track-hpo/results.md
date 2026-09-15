# Track 1 — results

## Best config vs baseline
Tiny data (120 imgs, random init, no pretrain): baseline verify test_acc 0.1667 ece 0.1336.
Max 12-param BOHB 6 trials best val_acc 0.2917 test_acc 0.25, beats tiny baseline by 0.125 val.
See `outputs/mobilenetv3_small/metrics/hpo/hpo_summary.json`.

## Pruner analysis
BOHB 4-param 9 trials: 9 complete 0 pruned best 0.2083 19.6s.
Hyperband 4-param 9 trials: 9 complete 0 pruned best 0.25 22.2s.
BOHB max 12-param 6 trials: 5 complete 1 pruned best 0.2917 18.0s.
Pruner only triggers on repeated rung with spread. Single final report per trial limits mid training kills.

## Platform notes (what the automated HPO must handle)
Sampler swap is one line: tpe for BOHB, random for Hyperband, same HyperbandPruner.
Max tunable without shared change is 12 keys: 4 continuous plus 8 bool flags.
trials.csv now logs test_accuracy test_loss epochs_trained budget for every trial including pruned.
Need per epoch reporting in train loop for true successive halving.
