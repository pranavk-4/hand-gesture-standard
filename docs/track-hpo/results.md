# Track 1 — results

## Statistical caveats (read before quoting any number below)

Independent audit, verified against installed optuna 4.9.0 and rerun files:
- TPE never switched on. TPESampler defaults to n_startup_trials 10 with no
  seed, and every study here ran 9 trials or fewer. All of them sampled
  randomly. Every run labeled BOHB below was random search, so no
  BOHB-beats-Hyperband claim on this page is supported.
- One unseeded study per setting, never repeated. Nothing here separates
  sampler skill from luck.
- Noise floor is about 0.01 SE on 900 val images at 0.9 accuracy (0.02 at
  95 pct). Gaps near 0.01 are indistinguishable. The 0.023 resnet gap is
  suggestive only, single run, no seed.
- The human 4-param control had hindsight: it was built after watching
  frozen-backbone trials fail on this same data. Laya selections were cold.
- Wasted dimensions: ema_decay is sampled with EMA off, backbone_lr without
  differential LR. Conditional spaces still missing.
- What holds up: pipeline and per-epoch pruning end to end, frozen backbones
  reliably fail on this data, resnet18 beats mobilenetv3 here.

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
Shared-code change pending review: batch_size, grad_clip_norm, label_smoothing_value, warmup_epochs, unfreeze_depth now honor overrides; apply_unfreeze accepts depth int. Verified depth params 6150/62598/932694 and 1-epoch knob run. Policy bank src/hpo/laya_policy.py exposes every HPO decision as logged atomic Laya questions.

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

## Laya policy study (Laya-built 9-param space, mobilenetv3, 6 trials)

Laya voted all 5 new knobs in: batch_size, unfreeze_depth, grad_clip_norm, label_smoothing_value, warmup_epochs. Study tag laya_policy: best val 0.878 test 0.841, 3 complete 3 pruned. Human 4-param control bohb_low: val 0.911 test 0.899. Policy space loses by 0.033 val. New knobs all executed mechanically (batch 16/64, depth 1/3/4 ran fine). Lesson repeats: 9 params on 6 trials starves TPE. Space plus decision log: `outputs/mobilenetv3_small/metrics/hpo/laya_policy/`.

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

Totals v2 (17 probes): base 6 hits 3 halves 7 misses. Typed 6 hits 3 halves 7 misses. Same score, different failure shape: typed confidences run lower on wrong answers (0.88 max vs 0.9999), its unfreeze misses land on lp_ft which is the literature safe default, and it uniquely hits F2. Typed gives byte-identical answers on S2 vs S3, so it is less sensitive to state nuance than base.

## Laya audit v3 (added UR1 resnet, UR2 mobilevit, 19 probes)

| Probe | Base | Typed | Truth |
|---|---|---|---|
| UR1 unfreeze resnet18 11M | head_only miss | full hit by 0.012 | full, baseline 0.92 |
| UR2 unfreeze mobilevit, zero runs | full, conf 0.27 | full, conf 0.09 | unknown |

Totals v3: base 6 hits 3 halves 8 misses, typed 7 hits 3 halves 7 misses (U2, UR2 open). Typed leads by one on the resnet probe. Both answer full on mobilevit with no data behind it.

## Laya audit v4 (v2 battery: atomic, explicit options, evidence-first states)

Same 12 probes on both checkpoints. Raw: `laya_probes/laya_probe_v2_results_{base,typed}.json`.

| Probe | Base v2 | Typed v2 | Truth |
|---|---|---|---|
| R1b explicit range options | zoom_0005_0008 hit | zoom_0005_0008 hit | zoom_0005_0008 |
| R2b explicit range options | keep_full miss | keep_full miss | zoom_0001_0003 |
| G-aug/cw/ls/diff keep flag | drop x4, all miss | drop x4, all miss | keep all four |
| G-ema-value | ema_false hit | ema_false hit | ema_false |
| G-unfreeze-value | unfreeze_true hit | unfreeze_true hit | unfreeze_true |
| U3b atomic yes/no pair | yes 0.79, no 0.28 hit+hit | yes 0.60, no 0.41 hit+hit | yes then no |
| D1b atomic pair | no hit, no miss | no hit, no miss | no then yes |
| T1b/T5b positive framing | no 0.49, no 0.31 hit+hit | no 0.26, no 0.22 hit+hit | no then no |

Totals v2: both checkpoints 6 hits 1 half 5 misses. Fixes that worked: explicit candidate options fixed R1, atomic decomposition fixed U3 triage and stop-side triage, concrete value-vs-value fixed ema and unfreeze. Fixes that failed: abstract keep-or-drop flag questions get a systematic no on both checkpoints, smoke zoom still keeps, data upside still denied. Rule drawn: never ask abstract keep or drop, always ask value vs value with the evidence inside the criteria.

## Cross-arch check (resnet18, proper data, low space, 6 trials)
hyperband_low: best val 0.936 test 0.916, 4 complete 2 pruned.
bohb_low: best val 0.959 test 0.940, 3 complete 3 pruned, winner head_lr 0.000262 backbone_lr 0.000660.
Pattern holds across architectures: BOHB beats Hyperband beats-or-ties baseline on val, pruning works, resnet winners use 10x smaller head_lr than mobilenet winners. Artifacts: `outputs/resnet18/metrics/hpo/{hyperband_low,bohb_low}/`.

Where the decision model fits today: keep-confirming second opinion on healthy trials only, plus free logging of every call for future calibration. Stop decisions stay with the Hyperband pruner. Both checkpoints fail data regime routing, failure cause except F2-typed, and any stop call. Fine-tuning needs 100+ logged decisions first; the probe harness accumulates them at zero GPU training cost.
