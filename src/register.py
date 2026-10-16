"""Rejestracja wytrenowanego modelu w MLflow Model Registry (wywoływane z Jenkinsa po progu jakości)."""
import argparse
import json
from pathlib import Path

import mlflow
import torch
import transformers
from mlflow import MlflowClient
from transformers import pipeline

from src.tracking import setup_mlflow

MODEL_NAME = "cryptobert-sentiment"


def register(model_dir: str = "models/model", reports_dir: str = "reports", alias: str = "champion") -> str:
    setup_mlflow()
    run_id = json.loads((Path(reports_dir) / "train_run.json").read_text())["run_id"]
    classifier = pipeline("text-classification", model=model_dir, tokenizer=model_dir, top_k=None)

    with mlflow.start_run(run_id=run_id):
        info = mlflow.transformers.log_model(
            transformers_model=classifier,
            artifact_path="model",
            task="text-classification",
            registered_model_name=MODEL_NAME,
            # jawna lista – domyślna wymaga torchvision, którego nie używamy
            pip_requirements=[f"torch=={torch.__version__.split('+')[0]}",
                              f"transformers=={transformers.__version__}"],
        )

    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version)
    print(f"Zarejestrowano {MODEL_NAME} w wersji {version} (alias: {alias})")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="champion")
    register(alias=parser.parse_args().alias)
