"""Etap `train`: dostrajanie CryptoBERT z logowaniem przebiegu do MLflow."""
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.params import load_params
from src.text import ID2LABEL, LABEL2ID
from src.tracking import flatten, setup_mlflow


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def history_to_frame(log_history: list[dict]) -> pd.DataFrame:
    """Jeden wiersz na epokę: strata treningowa i metryki walidacyjne."""
    rows = {}
    for entry in log_history:
        if "epoch" not in entry or "train_runtime" in entry:
            continue
        epoch = round(entry["epoch"])
        row = rows.setdefault(epoch, {"epoch": epoch})
        if "loss" in entry:
            row["train_loss"] = entry["loss"]
        for key in ("eval_loss", "eval_accuracy", "eval_f1"):
            if key in entry:
                row[key] = entry[key]
    return pd.DataFrame(list(rows.values()))


def train(params: dict, data_dir: str = "data/processed", model_dir: str | None = "models/model",
          reports_dir: str = "reports", run_name: str | None = None, nested: bool = False) -> dict:
    p = params["train"]
    tokenizer = AutoTokenizer.from_pretrained(p["base_model"])

    def load_split(name):
        df = pd.read_csv(Path(data_dir) / f"{name}.csv")
        dataset = Dataset.from_pandas(df[["text", "label"]])
        return dataset.map(lambda b: tokenizer(b["text"], truncation=True, max_length=p["max_length"]),
                           batched=True)

    train_dataset, val_dataset = load_split("train"), load_split("val")

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            p["base_model"], num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    setup_mlflow()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as checkpoints, \
            mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.log_params(flatten(params))
        mlflow.set_tag("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

        args = TrainingArguments(
            output_dir=checkpoints,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=p["learning_rate"],
            per_device_train_batch_size=p["batch_size"],
            per_device_eval_batch_size=32,
            num_train_epochs=p["epochs"],
            weight_decay=p["weight_decay"],
            lr_scheduler_type=p["lr_scheduler"],
            fp16=p.get("fp16", False) and torch.cuda.is_available(),
            seed=params["data"]["seed"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            report_to=["mlflow"],
        )
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            # patience 0 = bez early stoppingu (pełna liczba epok)
            callbacks=[EarlyStoppingCallback(early_stopping_patience=p["early_stopping_patience"])]
            if p["early_stopping_patience"] > 0 else [],
        )
        trainer.train()
        if trainer.state.global_step < trainer.state.max_steps:
            print(f"Early stopping po {round(trainer.state.epoch)} z {p['epochs']} epok")
        mlflow.log_metric("stopped_epoch", round(trainer.state.epoch))

        history = history_to_frame(trainer.state.log_history)
        best = trainer.evaluate()
        mlflow.log_metrics({"best_val_loss": best["eval_loss"], "best_val_f1": best["eval_f1"]})

        if model_dir:
            trainer.save_model(model_dir)
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
            history.to_csv(Path(reports_dir) / "train_history.csv", index=False)
            (Path(reports_dir) / "train_run.json").write_text(json.dumps({"run_id": run.info.run_id}))

    return {"run_id": run.info.run_id, "history": history, "best": best}


if __name__ == "__main__":
    train(load_params())
