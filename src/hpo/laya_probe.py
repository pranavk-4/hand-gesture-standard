"""Manual Laya decision probes vs human HPO decisions.

Each scenario is built from measured run history. Ground truth is known,
so every Laya answer can be scored against what actually worked.
Usage: python -m src.hpo.laya_probe (run from repo root, USE_TF=0).
"""

import json
import os

os.environ.setdefault("USE_TF", "0")

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
]


def main() -> None:
    import laya

    agent = laya.load("convaiinnovations/laya")
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
    with open("laya_probe_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("=" * 70)
    print("wrote laya_probe_results.json")


if __name__ == "__main__":
    main()
