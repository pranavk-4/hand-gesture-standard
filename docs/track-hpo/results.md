# Track 1 — results

## Best config vs baseline
Proper data 4200 train 900 val 900 test letterbox pretrained baseline 60 epochs: val 0.9022 test 0.8678.
Best is BOHB low 4-param 6 trials tag bohb_low trial 004: val 0.9111 test 0.8989.
Beats baseline by 0.009 val and 0.031 test.
Old smoke numbers are superseded and kept below for history.

## Pruner analysis
Proper baseline no HPO 60 epochs: val 0.9022 test 0.8678, 1843s.
Hyperband low 4-param partial 4 of 6 trials: trial_000 val 0.9056 test 0.8811 30ep, trial_001 val 0.7256 test 0.72 30ep, trial_002 val 0.74 test 0.7178 30ep, trial_003 val 0.8933 test 0.8756 25ep early stopped. Trial_004 cut by tool timeout, study csv not written.
Hyperband high 12-param 6 trials: 2 complete 4 pruned best val 0.8989 test 0.8733.
BOHB low 4-param 6 trials: 2 complete 4 pruned best val 0.9111 test 0.8989.
BOHB high 12-param 6 trials: 4 complete 2 pruned best val 0.9 test 0.8889.
BOHB beats Hyperband at both sizes. Per epoch pruning now prunes 2 to 4 trials per run.
History on smoke 1200 imgs 15 epochs: baseline val 0.4056 test 0.3889. Hyperband low best val 0.5166 test 0.4889. Hyperband high best val 0.3222 test 0.2944. BOHB low best val 0.55 test 0.5055. BOHB high best val 0.6 test 0.5889. High space only paid with TPE.

## Platform notes (what the automated HPO must handle)
Sampler swap is one line: tpe for BOHB, random for Hyperband, same HyperbandPruner.
Max tunable without shared change is 12 keys: 4 continuous plus 8 bool flags.
trials.csv now logs test_accuracy test_loss epochs_trained budget for every trial including pruned.
Per epoch pruning is live via epoch_callback in train.py with trial.report each epoch.
ONNX parity now passes with predicted class agreement on proper baseline.
Each tagged HPO run writes its own subdir under metrics hpo, so runs do not overwrite.

## Laya decision probes (zero-shot, local `convaiinnovations/laya`)

Probe script `src/hpo/laya_probe.py` asks 9 fixed scenarios built from measured history.
Install: `pip install laya`. Raw answers: `outputs/mobilenetv3_small/metrics/hpo/laya_probes/`.

| Probe | Laya answer | Human / truth | Hit |
|---|---|---|---|
| S1 space after smoke pilot | small (0.91, conf 0.55) | large won (0.60 vs 0.55) | miss |
| S2 space after proper pilot | small (0.89, conf 0.50) | small won (0.911 vs 0.90) | hit |
| S3 space after tiny pilot | small (0.81, conf 0.29) | neither, fix data first | half |
| U1 unfreeze proper data | full (0.28, conf 0.006) | full (test 0.868) | hit by 0.004 margin |
| U2 unfreeze tiny data | head_only (0.32, conf 0.01) | unknown, never tested staged | open |
| U3 unfreeze given frozen-trial evidence | head_only (0.32, conf 0.02) | full, unfrozen all beat 0.85 | miss |
| T1 triage diverged trial | keep (noul 0.0001, conf 0.9999) | stop, pruner killed it | miss, overconfident |
| T2 triage healthy trial | keep (noul 0.0) | keep, became best 0.906 | hit |
| T3 triage slow starter | keep (noul 0.0) | keep, reached 0.88 | hit |

Reading: choice answers collapse to one option (small, head_only) with near uniform spreads on 4-way unfreeze. Polarity check on T1 (stop framing vs keep framing) gives incoherent pair 0.0001 vs 0.307. Zero-shot Laya does not beat human ranges here. Matches the model card honest limit: base checkpoint near chance zero-shot, ships overconfident. Next step if pursued: log decisions with outcomes, refit temperature, or fine-tune on our trial histories before trusting it in the loop.
