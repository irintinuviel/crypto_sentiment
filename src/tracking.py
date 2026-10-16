import os

import mlflow
from dotenv import load_dotenv


def setup_mlflow(experiment: str | None = None) -> None:
    """Konfiguracja MLflow ze zmiennych środowiskowych lub pliku .env.

    Bez MLFLOW_TRACKING_URI przebiegi zapisują się lokalnie w ./mlruns.
    """
    load_dotenv()
    experiment = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "cryptobert-sentiment")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
    mlflow.set_experiment(experiment)


def flatten(params: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
