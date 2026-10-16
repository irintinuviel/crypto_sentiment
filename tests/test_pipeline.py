"""Test dymny całego potoku na malutkim modelu: prepare -> train -> evaluate -> próg jakości -> API."""
import json

import pytest
from fastapi.testclient import TestClient

from src.check_metrics import check
from src.data import prepare
from src.evaluate import compute_metrics, evaluate
from src.text import LABELS
from src.train import train

pytestmark = pytest.mark.slow


@pytest.fixture
def trained_model(params, tmp_path, mlflow_tmp):
    data_dir, model_dir, reports_dir = tmp_path / "processed", tmp_path / "model", tmp_path / "reports"
    prepare(params, out_dir=data_dir)
    result = train(params, data_dir=data_dir, model_dir=model_dir, reports_dir=reports_dir)
    return {"data_dir": data_dir, "model_dir": model_dir, "reports_dir": reports_dir, "result": result}


def test_train_and_evaluate(params, trained_model):
    reports = trained_model["reports_dir"]
    assert (trained_model["model_dir"] / "config.json").exists()
    assert list(trained_model["result"]["history"]["epoch"]) == [1]

    metrics = evaluate(params, data_dir=trained_model["data_dir"], model_dir=trained_model["model_dir"],
                       reports_dir=reports)
    assert 0 <= metrics["f1_weighted"] <= 1
    assert json.loads((reports / "metrics.json").read_text()) == metrics
    for name in ("confusion_matrix.png", "predictions.csv", "classification_report.txt", "train_run.json"):
        assert (reports / name).exists()

    assert check(reports / "metrics.json", "f1_weighted", 0.0)
    assert not check(reports / "metrics.json", "f1_weighted", 1.01)


def test_api_predict(trained_model, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained_model["model_dir"]))
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

        response = client.post("/predict", json={"texts": ["bitcoin to the moon", "eth crash"]})
        assert response.status_code == 200
        body = response.json()
        assert [p["text"] for p in body] == ["bitcoin to the moon", "eth crash"]
        for p in body:
            assert p["label"] in LABELS
            assert sum(p["scores"].values()) == pytest.approx(1, abs=1e-3)

        assert client.post("/predict", json={"texts": []}).status_code == 422
        assert "sentiment_predictions_total" in client.get("/metrics").text


def test_compute_metrics_perfect():
    metrics = compute_metrics(LABELS, LABELS)
    assert metrics["accuracy"] == metrics["f1_weighted"] == 1.0
