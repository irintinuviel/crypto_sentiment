"""Etap `prepare`: podział zbioru na część treningową, walidacyjną i testową."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.params import load_params
from src.text import LABEL2ID


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "pseudo_label"])
    df = df[df["pseudo_label"].isin(LABEL2ID)]
    df["label"] = df["pseudo_label"].map(LABEL2ID).astype(int)
    return df[["text", "label"]].reset_index(drop=True)


def split_dataset(df: pd.DataFrame, val_size: float, test_size: float, seed: int,
                  sample_size: int = 0) -> dict[str, pd.DataFrame]:
    if sample_size and sample_size < len(df):
        df, _ = train_test_split(df, train_size=sample_size, stratify=df["label"], random_state=seed)

    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  stratify=train_val["label"], random_state=seed)
    return {"train": train, "val": val, "test": test}


def prepare(params: dict, out_dir: str = "data/processed") -> dict[str, pd.DataFrame]:
    p = params["data"]
    df = load_dataset(p["input"])
    splits = split_dataset(df, p["val_size"], p["test_size"], p["seed"], p.get("sample_size", 0))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(Path(out_dir) / f"{name}.csv", index=False)
        print(f"{name}: {len(split)} próbek")
    return splits


if __name__ == "__main__":
    prepare(load_params())
