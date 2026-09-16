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


def run_hpo(cfg, n_trials: int | None = None, space_path: Path | None = None,
            sampler_name: str | None = None, tag: str | None = None) -> dict:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import RandomSampler, TPESampler

    from src.training.train import run_training

    space_path = space_path or (Path(__file__).parent / "search_space.yaml")
    space = _load_space(space_path)
    hb = space.get("hyperband", {})
    n_trials = n_trials or space.get("study", {}).get("n_trials", 27)
    sampler_name = sampler_name or space.get("study", {}).get("sampler", "tpe")
    min_r = hb.get("min_resource_epochs", 5)
    max_r = hb.get("max_resource_epochs", 30)

    pruner = HyperbandPruner(
        min_resource=min_r,
        max_resource=max_r,
        reduction_factor=hb.get("reduction_factor", 3),
    )
    if sampler_name == "random":
        sampler = RandomSampler()
    else:
        sampler = TPESampler()
    logger.info("HPO sampler=%s pruner=hyperband trials=%s space=%s", sampler_name, n_trials, space_path)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Per-run subdir so the 5 comparison cases (Hyperband/BOHB x low/high, +
    # baseline) don't overwrite each other's trials.csv / best_config.
    tag = tag or f"{sampler_name}__{space_path.stem}"
    hpo_dir = Path(cfg.output.metrics_dir) / "hpo" / tag
    hpo_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, space)

        # TRUE multi-fidelity: every trial runs up to max_resource epochs and
        # reports its val score each epoch; the HyperbandPruner cuts weak
        # trials at the successive-halving rungs. (The old code assigned a
        # fixed budget by trial number and reported only once at the end, so
        # 15-epoch trials always beat 5-epoch trials regardless of params.)
        def on_epoch(epoch: int, val_acc: float) -> None:
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        summary = run_training(cfg, run_name=f"hpo/{tag}/trial_{trial.number:03d}",
                               overrides=params, max_epochs_override=max_r,
                               epoch_callback=on_epoch)
        trial.set_user_attr("test_accuracy", summary["test_accuracy"])
        trial.set_user_attr("test_loss", summary["test_loss"])
        trial.set_user_attr("epochs_trained", summary["epochs_trained"])
        trial.set_user_attr("budget", max_r)
        return summary["best_val_accuracy"]

    study.optimize(objective, n_trials=n_trials)

    rows = [{
        "trial": t.number,
        "state": t.state.name,
        "value": t.value,
        **t.params,
        "test_accuracy": t.user_attrs.get("test_accuracy"),
        "test_loss": t.user_attrs.get("test_loss"),
        "epochs_trained": t.user_attrs.get("epochs_trained"),
        "budget": t.user_attrs.get("budget"),
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
        "tag": tag,
        "n_trials": len(study.trials),
        "n_complete": sum(1 for t in study.trials if str(t.state) == "TrialState.COMPLETE"),
        "n_pruned": sum(1 for t in study.trials if str(t.state) == "TrialState.PRUNED"),
        "best_value": study.best_value if study.best_trial else None,
        "best_params": best_params,
        "sampler": sampler_name,
        "space": space_path.stem,
        "min_resource_epochs": min_r,
        "max_resource_epochs": max_r,
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
    ap.add_argument("--space", default=None)
    ap.add_argument("--sampler", choices=["tpe", "random"], default=None)
    ap.add_argument("--tag", default=None,
                    help="Output subdir name under metrics/hpo/ (default: <sampler>__<space>).")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / "configs" / cfg_path.name
    cfg = load_config(cfg_path)
    space_path = Path(args.space) if args.space else None
    run_hpo(cfg, n_trials=args.trials, space_path=space_path, sampler_name=args.sampler, tag=args.tag)
