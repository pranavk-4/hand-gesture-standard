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

## Laya audit v2 (17 probes x base vs typed-decisions checkpoints)

Extended battery adds range zoom, trial budget, failure cause, data regime routing, mid-training triage. Raw answers: `outputs/mobilenetv3_small/metrics/hpo/laya_probes/laya_probe_results_{base,typed}.json`. Score counts U2 as open (staged schedules never tested on tiny data).

| Probe | Base | Typed | Truth |
|---|---|---|---|
| S1 space smoke | small miss | small miss | large won |
| S2 space proper | small hit | small hit | small won |
| S3 space tiny | small half | small half | fix data, not space |
| U1 unfreeze proper | full hit by 0.004 | lp_ft miss | full |
| U3 unfreeze evidence | head_only miss | lp_ft miss | full |
| R1 range proper | keep hit | keep hit | zoom or keep |
| R2 range smoke | keep half | keep half | zoom_in |
| B1 budget 12-param | trials_6 half | trials_6 half | 20 preferred, 6 sufficed |
| F1 onnx failure | export_bug miss | export_bug miss | tolerance_too_strict |
| F2 frozen trial | bad_lr miss | frozen_backbone hit by 0.009 | frozen_backbone |
| D1 data regime | more_trials miss | more_trials miss | more_data |
| T1 triage diverged | keep miss, conf 0.9999 | keep miss, noul 0.12 | stop |
| T2/T3/T4 triage keep | keep hit x3 | keep hit x3 | keep |
| T5 triage weak | keep miss, conf 0.9999 | keep miss, noul 0.12 | stop |

Totals: base 6 hits 3 halves 7 misses. Typed 6 hits 3 halves 7 misses. Same score, different failure shape: typed confidences run lower on wrong answers (0.88 max vs 0.9999), its unfreeze misses land on lp_ft which is the literature safe default, and it uniquely hits F2. Typed gives byte-identical answers on S2 vs S3, so it is less sensitive to state nuance than base.

Where the decision model fits today: keep-confirming second opinion on healthy trials only, plus free logging of every call for future calibration. Stop decisions stay with the Hyperband pruner. Both checkpoints fail data regime routing, failure cause except F2-typed, and any stop call. Fine-tuning needs 100+ logged decisions first; the probe harness accumulates them at zero GPU training cost.
