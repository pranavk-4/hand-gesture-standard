"""Manual Laya decision probes vs human HPO decisions.

Each scenario is built from measured run history. Ground truth is known,
so every Laya answer can be scored against what actually worked.
Usage: python -m src.hpo.laya_probe (run from repo root, USE_TF=0).
"""

import argparse
import json
import os

os.environ.setdefault("USE_TF", "0")

CHECKPOINTS = {
    "base": {"model_id": "convaiinnovations/laya"},
    "typed": {"model_id": "convaiinnovations/laya", "subfolder": "typed-decisions"},
}

SCENARIOS = [
    {
        "id": "S1-space-smoke",
        "kind": "space",
        "state": (
            "Pilot HPO on 1200 hand gesture images, 6 classes, mobilenetv3, pretrained. "
            "BOHB low space with 4 params over 6 trials. Best val accuracy 0.55. "
            "Best head_lr 0.0028 sits mid range of 0.0001 to 0.01. "
            "Technique flags unfreeze, augmentation, label smoothing were never varied."
        ),
        "questions": {
            "next_space": {
                "type": "choice",
                "instructions": "Which search space should the next HPO round use?",
                "criteria": {
                    "small": "keep the 4-param space of learning rates and decay",
                    "large": "expand to the 12-param space adding 8 technique flags",
                },
            }
        },
        "human": "large",
        "truth": "large won: BOHB high reached val 0.60 vs 0.55",
    },
    {
        "id": "S2-space-proper",
        "kind": "space",
        "state": (
            "Pilot HPO on 4200 hand gesture images, 6 classes, mobilenetv3, pretrained. "
            "BOHB low space with 4 params over 6 trials. Best val accuracy 0.911. "
            "Best head_lr 0.0071 sits near the top of the 0.0001 to 0.01 range. "
            "Technique flags were never varied."
        ),
        "questions": {
            "next_space": {
                "type": "choice",
                "instructions": "Which search space should the next HPO round use?",
                "criteria": {
                    "small": "keep the 4-param space of learning rates and decay",
                    "large": "expand to the 12-param space adding 8 technique flags",
                },
            }
        },
        "human": "large",
        "truth": "small won: BOHB high reached val 0.90 vs 0.911",
    },
    {
        "id": "S3-space-tiny",
        "kind": "space",
        "state": (
            "Pilot HPO on 120 tiny synthetic images, 6 classes, mobilenetv3, random init no pretrain. "
            "9 trials. Best val accuracy 0.25, barely above chance 0.167. "
            "All trials score the same whatever the learning rate."
        ),
        "questions": {
            "next_space": {
                "type": "choice",
                "instructions": "Which search space should the next HPO round use?",
                "criteria": {
                    "small": "keep the 4-param space of learning rates and decay",
                    "large": "expand to the 12-param space adding 8 technique flags",
                },
            }
        },
        "human": "neither, fix data first",
        "truth": "neither won: real data lift beat any space change, 0.25 to 0.87 test",
    },
    {
        "id": "U1-unfreeze-proper",
        "kind": "unfreeze",
        "state": (
            "Image classification, 4200 training images, 6 hand gesture classes, "
            "mobilenetv3_small_100 with 2.5M params pretrained on ImageNet. "
            "Medium domain gap: natural images to hands. Small model, sufficient data."
        ),
        "questions": {
            "schedule": {
                "type": "choice",
                "instructions": "Which backbone unfreeze schedule should training use?",
                "criteria": {
                    "head_only": "freeze backbone, train classifier head only",
                    "lp_ft": "train head first, then unfreeze all layers with low LR",
                    "gradual": "unfreeze one block group at a time from head to stem",
                    "full": "fine-tune all layers from the start with differential LRs",
                },
            }
        },
        "human": "full",
        "truth": "full reached test 0.868; frozen trials scored 0.2 to 0.35",
    },
    {
        "id": "U2-unfreeze-tiny",
        "kind": "unfreeze",
        "state": (
            "Image classification, 120 training images, 6 hand gesture classes, "
            "mobilenetv3_small_100 pretrained on ImageNet. Tiny data, high overfit risk."
        ),
        "questions": {
            "schedule": {
                "type": "choice",
                "instructions": "Which backbone unfreeze schedule should training use?",
                "criteria": {
                    "head_only": "freeze backbone, train classifier head only",
                    "lp_ft": "train head first, then unfreeze all layers with low LR",
                    "gradual": "unfreeze one block group at a time from head to stem",
                    "full": "fine-tune all layers from the start with differential LRs",
                },
            }
        },
        "human": "full was used, scored 0.25",
        "truth": "unknown: staged schedules never tested on tiny data",
    },
    {
        "id": "U3-unfreeze-frozen-evidence",
        "kind": "unfreeze",
        "state": (
            "Measured HPO evidence on 4200 images: every trial with frozen backbone "
            "scored val 0.2 to 0.35 and was pruned early. Every trial with unfrozen "
            "backbone scored above 0.85. Backbone gradient norm reads exactly 0 when frozen."
        ),
        "questions": {
            "schedule": {
                "type": "choice",
                "instructions": "Which backbone unfreeze schedule should the next round use?",
                "criteria": {
                    "head_only": "freeze backbone, train classifier head only",
                    "lp_ft": "train head first, then unfreeze all layers with low LR",
                    "gradual": "unfreeze one block group at a time from head to stem",
                    "full": "fine-tune all layers from the start with differential LRs",
                },
            }
        },
        "human": "full",
        "truth": "full: unfrozen trials all beat 0.85",
    },
    {
        "id": "T1-triage-diverged",
        "kind": "triage",
        "state": (
            "Trial val accuracy by epoch: 0.19, 0.20, 0.23, 0.24, 0.27. "
            "Backbone frozen, backbone gradient norm 0. "
            "Leader after same epochs is above 0.70. Budget is 30 epochs."
        ),
        "questions": {
            "stop_now": {
                "type": "noul",
                "instructions": "Should this trial be stopped now to save GPU time?",
            }
        },
        "human": "stop",
        "truth": "stopped: pruner killed it, finished 0.35",
    },
    {
        "id": "T2-triage-healthy",
        "kind": "triage",
        "state": (
            "Trial val accuracy by epoch: 0.44, 0.60, 0.51, 0.48, 0.55. "
            "Backbone unfrozen with healthy gradient norms near 10. "
            "Matches the leader trajectory. Budget is 30 epochs."
        ),
        "questions": {
            "stop_now": {
                "type": "noul",
                "instructions": "Should this trial be stopped now to save GPU time?",
            }
        },
        "human": "keep",
        "truth": "kept: became best trial at val 0.906",
    },
    {
        "id": "T3-triage-slow-start",
        "kind": "triage",
        "state": (
            "Trial val accuracy by epoch: 0.18, 0.30, 0.52, 0.66, 0.74. "
            "Backbone unfrozen. Steep improvement each epoch. Budget is 30 epochs."
        ),
        "questions": {
            "stop_now": {
                "type": "noul",
                "instructions": "Should this trial be stopped now to save GPU time?",
            }
        },
        "human": "keep",
        "truth": "kept: reached val 0.88",
    },
    {
        "id": "R1-range-proper",
        "kind": "range",
        "state": (
            "Pilot HPO done. Best head_lr 0.0071 sits near the top of the "
            "0.0001 to 0.01 range. All 6 winners lie between 0.005 and 0.008. "
            "No trial hit the 0.01 ceiling."
        ),
        "questions": {
            "range_move": {
                "type": "choice",
                "instructions": "How should the head_lr range change for the next round?",
                "criteria": {
                    "zoom_in": "narrow around 0.005 to 0.008 winners",
                    "widen_up": "extend the top above 0.01",
                    "keep": "leave the range unchanged",
                },
            }
        },
        "human": "zoom_in or keep",
        "truth": "zoom_in or keep: best inside range, ceiling never binding",
    },
    {
        "id": "R2-range-smoke",
        "kind": "range",
        "state": (
            "Pilot HPO done on 1200 images. Best head_lr 0.0017 sits mid range "
            "of 0.0001 to 0.01. Results spread wide across the whole range."
        ),
        "questions": {
            "range_move": {
                "type": "choice",
                "instructions": "How should the head_lr range change for the next round?",
                "criteria": {
                    "zoom_in": "narrow around the 0.0017 winner",
                    "widen_up": "extend the top above 0.01",
                    "keep": "leave the range unchanged",
                },
            }
        },
        "human": "zoom_in",
        "truth": "zoom_in: winner mid range with wide spread",
    },
    {
        "id": "B1-budget",
        "kind": "budget",
        "state": (
            "Next round uses the 12-param space: 4 continuous plus 8 technique flags. "
            "Each full trial costs 15 GPU minutes. TPE sampler with Hyperband pruner."
        ),
        "questions": {
            "trial_budget": {
                "type": "choice",
                "instructions": "How many trials should the next round run?",
                "criteria": {
                    "trials_6": "6 trials, about 90 GPU minutes",
                    "trials_20": "20 trials, about 5 GPU hours",
                },
            }
        },
        "human": "trials_6 pilot then more",
        "truth": "trials_20 preferred by 10-per-param rule; trials_6 sufficed for TPE pilot",
    },
    {
        "id": "F1-onnx-failure",
        "kind": "failure",
        "state": (
            "Baseline trained fine: val 0.406, test 0.389, healthy curves. "
            "ONNX export parity check failed with max abs error 0.011 over the 1e-4 limit."
        ),
        "questions": {
            "cause": {
                "type": "choice",
                "instructions": "What caused the ONNX parity failure?",
                "criteria": {
                    "needs_retrain": "the trained model is bad and must retrain",
                    "export_bug": "the export code wrote a wrong graph",
                    "tolerance_too_strict": "raw logit noise trips an over strict threshold",
                },
            }
        },
        "human": "tolerance_too_strict",
        "truth": "tolerance_too_strict: fixed by predicted class agreement check",
    },
    {
        "id": "F2-frozen-trial",
        "kind": "failure",
        "state": (
            "Trial finished val 0.2 near chance. Backbone gradient norm exactly 0 "
            "every epoch. Learning rate 0.002 inside normal range. Full 30 epoch budget used."
        ),
        "questions": {
            "cause": {
                "type": "choice",
                "instructions": "What caused this trial to fail?",
                "criteria": {
                    "frozen_backbone": "backbone was frozen so only the head learned",
                    "bad_lr": "the learning rate was wrong for this trial",
                    "low_budget": "the epoch budget was too small",
                },
            }
        },
        "human": "frozen_backbone",
        "truth": "frozen_backbone: zero backbone grad with full budget and sane LR",
    },
    {
        "id": "D1-data-regime",
        "kind": "routing",
        "state": (
            "Pilot HPO on 120 tiny images, random init, 9 trials. Best val 0.25 "
            "barely above chance 0.167. All learning rates score the same."
        ),
        "questions": {
            "next_move": {
                "type": "choice",
                "instructions": "What is the highest leverage next move?",
                "criteria": {
                    "more_trials": "run many more HPO trials on the same data",
                    "more_data": "collect real images and rerun",
                    "bigger_model": "switch to a larger architecture",
                },
            }
        },
        "human": "more_data",
        "truth": "more_data: real data lift 0.25 to 0.87 beat any tuning",
    },
    {
        "id": "T4-triage-mid-healthy",
        "kind": "triage",
        "state": (
            "Trial val accuracy by epoch 6 to 10: 0.80, 0.75, 0.82, 0.81, 0.83. "
            "Backbone unfrozen with healthy gradient norms near 25. Tracks the leader."
        ),
        "questions": {
            "stop_now": {
                "type": "noul",
                "instructions": "Should this trial be stopped now to save GPU time?",
            }
        },
        "human": "keep",
        "truth": "kept: finished val 0.893",
    },
    {
        "id": "T5-triage-mid-weak",
        "kind": "triage",
        "state": (
            "Trial val accuracy at epoch 5 is 0.42 while the leader is above 0.75. "
            "Backbone gradient norm is 0. Budget is 30 epochs."
        ),
        "questions": {
            "stop_now": {
                "type": "noul",
                "instructions": "Should this trial be stopped now to save GPU time?",
            }
        },
        "human": "stop",
        "truth": "stopped: frozen trial family all finished near 0.2 to 0.35",
    },
    {
        "id": "UR1-unfreeze-resnet",
        "kind": "unfreeze",
        "state": (
            "Image classification, 4200 training images, 6 hand gesture classes, "
            "resnet18 with 11M params pretrained on ImageNet. "
            "Medium domain gap: natural images to hands. Larger model than mobilenet, same sufficient data. "
            "Unfrozen baseline reached val 0.928 test 0.92. Sister mobilenet runs show frozen trials stall at 0.2 to 0.35."
        ),
        "questions": {
            "schedule": {
                "type": "choice",
                "instructions": "Which backbone unfreeze schedule should training use?",
                "criteria": {
                    "head_only": "freeze backbone, train classifier head only",
                    "lp_ft": "train head first, then unfreeze all layers with low LR",
                    "gradual": "unfreeze one block group at a time from head to stem",
                    "full": "fine-tune all layers from the start with differential LRs",
                },
            }
        },
        "human": "full",
        "truth": "full: unfrozen baseline 0.92, mobilenet pattern transfers",
    },
    {
        "id": "UR2-unfreeze-mobilevit",
        "kind": "unfreeze",
        "state": (
            "Image classification, 4200 training images, 6 hand gesture classes, "
            "mobilevit_xxs hybrid vision transformer pretrained on ImageNet. "
            "No runs exist yet for this architecture. Transformers are data hungry "
            "and sensitive to full fine-tuning on small data."
        ),
        "questions": {
            "schedule": {
                "type": "choice",
                "instructions": "Which backbone unfreeze schedule should training use?",
                "criteria": {
                    "head_only": "freeze backbone, train classifier head only",
                    "lp_ft": "train head first, then unfreeze all layers with low LR",
                    "gradual": "unfreeze one block group at a time from head to stem",
                    "full": "fine-tune all layers from the start with differential LRs",
                },
            }
        },
        "human": "lp_ft or gradual per textbooks",
        "truth": "unknown: zero mobilevit runs exist",
    },
]


def main() -> None:
    import laya

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=["base", "typed"], default="base")
    args = ap.parse_args()
    spec = CHECKPOINTS[args.checkpoint]
    agent = laya.load(spec["model_id"], **{k: v for k, v in spec.items() if k != "model_id"})
    out_name = f"laya_probe_results_{args.checkpoint}.json"
    results = []
    for sc in SCENARIOS:
        out = agent.predict(sc["state"], sc["questions"])
        row = {"id": sc["id"], "kind": sc["kind"], "human": sc["human"],
               "truth": sc["truth"], "answers": out["answers"]}
        results.append(row)
        print("=" * 70)
        print(row["id"], "| human:", row["human"])
        print("truth:", row["truth"])
        print(json.dumps(row["answers"], indent=1))
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("=" * 70)
    print("wrote", out_name)


if __name__ == "__main__":
    main()
