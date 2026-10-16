"""Etap `evaluate`: metryki na zbiorze testowym, macierz pomyłek i błędnie sklasyfikowane próbki."""
import json
from pathlib import Path

import matplotlib
import mlflow
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from src.params import load_params
from src.predict import Predictor
from src.text import LABELS
from src.tracking import setup_mlflow

matplotlib.use("Agg")


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    metrics.update({f"f1_{label}": score for label, score in zip(LABELS, per_class)})
    return {k: round(float(v), 4) for k, v in metrics.items()}


def save_confusion_matrix(y_true, y_pred, path: Path) -> None:
    import matplotlib.pyplot as plt

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=LABELS, cmap="Blues",
                                                   values_format="d")
    disp.ax_.set_title("Macierz pomyłek – zbiór testowy")
    disp.ax_.set_xlabel("Przewidziana etykieta")
    disp.ax_.set_ylabel("Rzeczywista etykieta")
    disp.figure_.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(disp.figure_)


def evaluate(params: dict, data_dir: str = "data/processed", model_dir: str = "models/model",
             reports_dir: str = "reports") -> dict:
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(Path(data_dir) / "test.csv")
    predictor = Predictor(model_dir, max_length=params["train"]["max_length"])
    predictions = predictor.predict(test["text"].tolist())

    results = pd.DataFrame({
        "text": test["text"],
        "actual": [LABELS[i] for i in test["label"]],
        "predicted": [p["label"] for p in predictions],
    })
    metrics = compute_metrics(results["actual"], results["predicted"])
    report = classification_report(results["actual"], results["predicted"], labels=LABELS, digits=4,
                                   zero_division=0)

    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "classification_report.txt").write_text(report)
    results[["actual", "predicted"]].to_csv(reports / "predictions.csv", index=False)
    results[results["actual"] != results["predicted"]].to_csv(reports / "misclassified.csv", index=False)
    save_confusion_matrix(results["actual"], results["predicted"], reports / "confusion_matrix.png")

    # dopisanie wyników do przebiegu MLflow z etapu treningu
    setup_mlflow()
    run_file = reports / "train_run.json"
    run_id = json.loads(run_file.read_text())["run_id"] if run_file.exists() else None
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        for name in ("metrics.json", "classification_report.txt", "confusion_matrix.png",
                     "misclassified.csv", "train_history.csv"):
            if (reports / name).exists():
                mlflow.log_artifact(str(reports / name), artifact_path="reports")

    print(report)
    return metrics


if __name__ == "__main__":
    evaluate(load_params())
