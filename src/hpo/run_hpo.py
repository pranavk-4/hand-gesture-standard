"""HPO runner — owned by Track 1 (Hyperband / BOHB teammate).

Implements BOHB-as-practiced: Optuna TPE sampler (Bayesian) + Hyperband
pruner (multi-fidelity successive halving). Each trial calls the shared
run_training() with `overrides` + `max_epochs_override` budget, so HPO
results are directly comparable to the frozen baseline.

Requires: pip install optuna>=3.0

Usage:
    python -m src.hpo.run_hpo --config configs/baseline_6class_mobilenetv3.yaml --trials 27

Outputs (all under outputs/<arch>/metrics/hpo/):
    trials.csv          every trial: params + value + pruned/complete
    best_config.yaml    overrides dict of the best trial
    hpo_summary.json    best value, n_pruned, wall clock
Failure contract: every pruned/failed trial is a ROW, not a deleted log.
The platform test is about failure patterns (pruner aggression, OOMs),
so do not filter trials.csv.
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_space(space_path: Path) -> dict:
    import yaml
    with open(space_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _suggest(trial, space: dict) -> dict:
    params = {}
    for name, spec in space["search_space"].items():
        t = spec["type"]
        if t == "log_float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif t == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search type {t!r} for {name}")
    return params


def run_hpo(cfg, n_trials: int | None = None, space_path: Path | None = None) -> dict:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler

    from src.training.train import run_training

    space_path = space_path or (Path(__file__).parent / "search_space.yaml")
    space = _load_space(space_path)
    hb = space.get("hyperband", {})
    n_trials = n_trials or space.get("study", {}).get("n_trials", 27)

    pruner = HyperbandPruner(
        min_resource=hb.get("min_resource_epochs", 5),
        max_resource=hb.get("max_resource_epochs", 30),
        reduction_factor=hb.get("reduction_factor", 3),
    )
    study = optuna.create_study(direction="maximize", sampler=TPESampler(), pruner=pruner)

    hpo_dir = Path(cfg.output.metrics_dir) / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, space)
        # Multi-fidelity: Hyperband allocates the epoch budget per bracket.
        # Map trial number -> budget rung (simple round-robin over rungs).
        max_r = hb.get("max_resource_epochs", 30)
        min_r = hb.get("min_resource_epochs", 5)
        eta = hb.get("reduction_factor", 3)
        rungs = []
        r = max_r
        while r >= min_r:
            rungs.append(r)
            r //= eta
        budget = rungs[trial.number % len(rungs)]
        summary = run_training(cfg, run_name=f"hpo_trial_{trial.number:03d}",
                               overrides=params, max_epochs_override=budget)
        # Report intermediate values so Hyperband can prune mid-training.
        # (Simplification: report final value at its budget step.)
        trial.report(summary["best_val_accuracy"], step=budget)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return summary["best_val_accuracy"]

    study.optimize(objective, n_trials=n_trials)

    rows = [{
        "trial": t.number,
        "state": t.state.name,
        "value": t.value,
        **t.params,
    } for t in study.trials]
    with open(hpo_dir / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["trial"])
        w.writeheader()
        w.writerows(rows)

    import yaml
    best_params = dict(study.best_trial.params) if study.best_trial else {}
    with open(hpo_dir / "best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"overrides": best_params}, f)

    summary = {
        "n_trials": len(study.trials),
        "n_complete": sum(1 for t in study.trials if str(t.state) == "TrialState.COMPLETE"),
        "n_pruned": sum(1 for t in study.trials if str(t.state) == "TrialState.PRUNED"),
        "best_value": study.best_value if study.best_trial else None,
        "best_params": best_params,
        "wall_clock_seconds": time.time() - t0,
    }
    with open(hpo_dir / "hpo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("HPO done: %s", summary)
    return summary


if __name__ == "__main__":
    from src.config import PROJECT_ROOT, load_config
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline_6class_mobilenetv3.yaml")
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / "configs" / cfg_path.name
    cfg = load_config(cfg_path)
    run_hpo(cfg, n_trials=args.trials)
