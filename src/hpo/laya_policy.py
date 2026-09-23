"""Laya policy bank: every HPO decision as an askable atomic question.

Covers the full decision surface so an agent never falls back to a fixed
human range: param inclusion, range presets, trials, brackets, sampler,
unfreeze schedule and depth, stop/keep, failure cause, data regime routing.
Each answer logs to JSONL with its state hash for later calibration.

Usage:
    from src.hpo.laya_policy import Policy
    pol = Policy(log_path="outputs/policy_log.jsonl")
    space = pol.ask_space(model_profile, candidates="all")
"""

import hashlib
import json
import os

os.environ.setdefault("USE_TF", "0")

PARAM_ROLES = {
    "head_lr": "step size for the classifier head, highest leverage",
    "backbone_lr": "step size for pretrained backbone, destroys features if high",
    "weight_decay": "L2 regularization, secondary effect",
    "ema_decay": "weight averaging memory, matters only if ema is on",
    "label_smoothing_value": "target softness value 0.0 to 0.2",
    "warmup_epochs": "warmup length 0 to 5",
    "grad_clip_norm": "caps backbone gradient blow-up",
    "batch_size": "batch noise and speed, rebuilds loaders",
    "unfreeze_backbone": "whole backbone on or head only",
    "unfreeze_depth": "how many trailing block groups to unfreeze, 0 is head only",
    "augmentation": "heavy augmentation on or off",
    "class_weighted_loss": "class weights on or off",
    "label_smoothing": "smoothing on or off",
    "ema": "weight averaging on or off",
    "differential_lr": "separate backbone LR or single LR",
    "cosine_schedule": "cosine annealing or constant LR",
    "warmup": "LR warmup on or off",
}

SCHEDULES = {
    "head_only": "freeze backbone, train classifier head only",
    "lp_ft": "train head first, then unfreeze all with low LR",
    "gradual": "unfreeze one block group at a time from head to stem",
    "full": "fine-tune all layers from the start with differential LRs",
}


class Policy:
    def __init__(self, model_id="convaiinnovations/laya", log_path=None):
        import laya
        self.agent = laya.load(model_id)
        self.log_path = log_path

    def _log(self, row: dict) -> None:
        row["state_hash"] = hashlib.sha256(
            row["state"].encode("utf-8")).hexdigest()[:12]
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def _ask(self, kind: str, state: str, questions: dict) -> dict:
        answers = self.agent.predict(state, questions)["answers"]
        self._log({"kind": kind, "state": state, "answers": answers})
        return answers

    def ask_include(self, profile: str, param: str) -> bool:
        a = self._ask("include", f"{profile} Candidate: {param}. "
                      f"Role: {PARAM_ROLES[param]}. Include it only if its "
                      f"measured effect beats its trial cost.",
                      {"include": {"type": "noul", "instructions":
                                   f"Should {param} go into this HPO space?"}})
        return a["include"]["noul"] >= 0.5

    def ask_preset(self, profile: str, param: str, presets: dict) -> str:
        criteria = {k: f"preset {k}: {v}" for k, v in presets.items()}
        a = self._ask("preset", f"{profile} {param} is included. Pick its range.",
                      {"preset": {"type": "choice",
                                  "instructions": f"Which range preset for {param}?",
                                  "criteria": criteria}})
        return a["preset"]["choice"]

    def ask_study(self, profile: str, trials: dict, brackets: dict) -> dict:
        a = self._ask("study", profile + " Pick study size for an overnight run.",
                      {"trials": {"type": "choice",
                                  "instructions": "How many trials?",
                                  "criteria": {k: f"{v} trials" for k, v in trials.items()}},
                       "brackets": {"type": "choice",
                                    "instructions": "Which epoch brackets?",
                                    "criteria": {k: f"epochs {v}" for k, v in brackets.items()}},
                       "sampler": {"type": "choice",
                                   "instructions": "Which sampler?",
                                   "criteria": {"bohb": "TPE plus Hyperband pruner",
                                              "hyperband": "random plus Hyperband pruner"}}})
        return {k: v["choice"] for k, v in a.items()}

    def ask_schedule(self, profile: str) -> str:
        a = self._ask("schedule", profile,
                      {"schedule": {"type": "choice",
                                    "instructions": "Which unfreeze schedule?",
                                    "criteria": SCHEDULES}})
        return a["schedule"]["choice"]

    def ask_stop(self, early_curve: str) -> bool:
        a = self._ask("triage", early_curve,
                      {"finish": {"type": "noul", "instructions":
                                  "Will this trial finish above its mid-training mark?"}})
        return a["finish"]["noul"] < 0.5

    def ask_failure(self, symptom: str, options: dict) -> str:
        a = self._ask("failure", symptom,
                      {"cause": {"type": "choice",
                                 "instructions": "What caused this failure?",
                                 "criteria": options}})
        return a["cause"]["choice"]

    def ask_routing(self, regime: str) -> str:
        a = self._ask("routing", regime,
                      {"move": {"type": "choice",
                                "instructions": "Highest leverage next move?",
                                "criteria": {"more_trials": "more HPO trials, same data",
                                           "more_data": "collect real images, rerun",
                                           "bigger_model": "switch architecture"}}})
        return a["move"]["choice"]
