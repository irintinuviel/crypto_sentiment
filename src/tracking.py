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
    print(f"MLflow: {mlflow.get_tracking_uri()} (eksperyment: {experiment})")
    try:
        mlflow.set_experiment(experiment)
    except mlflow.exceptions.MlflowException as e:
        raise RuntimeError(
            f"Brak dostępu do MLflow pod {mlflow.get_tracking_uri()} – sprawdź MLFLOW_TRACKING_URI "
            "(https://dagshub.com/<login>/<repozytorium>.mlflow) oraz login i token"
        ) from e


def flatten(params: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
