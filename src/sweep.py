"""Poszukiwanie hiperparametrów (Optuna, odpowiednik sweepa W&B) – każda próba to przebieg MLflow.

Użycie:
    python -m src.sweep --trials 9 --apply   # --apply zapisuje najlepsze parametry do params.yaml
"""
import argparse

import mlflow
import optuna

from src.params import load_params, override, save_params
from src.tracking import setup_mlflow
from src.train import train


def objective_factory(params: dict, data_dir: str):
    def objective(trial: optuna.Trial) -> float:
        trial_params = override(params, [
            f"train.learning_rate={trial.suggest_float('learning_rate', 1e-6, 5e-5, log=True)}",
            f"train.batch_size={trial.suggest_categorical('batch_size', [16, 32])}",
            f"train.weight_decay={trial.suggest_float('weight_decay', 0.0, 0.1)}",
            f"train.lr_scheduler={trial.suggest_categorical('lr_scheduler', ['linear', 'cosine'])}",
        ])
        result = train(trial_params, data_dir=data_dir, model_dir=None, run_name=f"trial-{trial.number}",
                       nested=True)
        return result["best"]["eval_f1"]

    return objective


def run_sweep(params: dict, n_trials: int = 9, data_dir: str = "data/processed") -> dict:
    setup_mlflow()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    with mlflow.start_run(run_name="sweep"):
        study.optimize(objective_factory(params, data_dir), n_trials=n_trials)
        mlflow.log_params({f"best.{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_f1", study.best_value)
    print(f"Najlepsze val F1 = {study.best_value:.4f}: {study.best_params}")
    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    params = load_params()
    best = run_sweep(params, args.trials)
    if args.apply:
        save_params(override(params, [f"train.{k}={v}" for k, v in best.items()]))
