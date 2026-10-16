import random

import pandas as pd
import pytest

TINY_MODEL = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"

WORDS = {
    "positive": ["moon", "pump", "gains", "bullish", "great"],
    "neutral": ["holding", "chart", "today", "wallet", "fees"],
    "negative": ["crash", "dump", "scam", "bearish", "rekt"],
}


@pytest.fixture
def dataset_csv(tmp_path):
    rng = random.Random(0)
    rows = [
        {"text": f"bitcoin is {' '.join(rng.choices(words, k=4))}", "pseudo_label": label}
        for label, words in WORDS.items()
        for _ in range(40)
    ]
    rows += [{"text": None, "pseudo_label": "positive"}, {"text": "eth ok", "pseudo_label": "unknown"}]
    path = tmp_path / "final_crypto_data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def params(dataset_csv):
    return {
        "data": {"input": str(dataset_csv), "val_size": 0.1, "test_size": 0.1, "seed": 42, "sample_size": 0},
        "train": {
            "base_model": TINY_MODEL, "max_length": 32, "learning_rate": 1e-3, "batch_size": 8, "epochs": 1,
            "weight_decay": 0.0, "lr_scheduler": "linear", "early_stopping_patience": 1, "fp16": False,
        },
    }


@pytest.fixture
def mlflow_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
