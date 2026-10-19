"""Etap `relabel`: ponowne pseudo-etykietowanie zbioru głosowaniem z VADER rozszerzonym o slang krypto.

Teksty pozostają te same (data/final_crypto_data.csv z DVC), zmieniają się tylko etykiety. Raport
reports/relabel.json pokazuje rozkład klas przed i po oraz ile etykiet się zmieniło.
"""
import json
from pathlib import Path

import pandas as pd

from src.dataset import pseudo_label
from src.params import load_params


def relabel(params: dict, reports_dir: str = "reports") -> pd.DataFrame:
    p = params["relabel"]
    df = pd.read_csv(p["input"]).dropna(subset=["text"])
    labeled = pseudo_label(df[["text"]], batch_size=p["batch_size"])
    labeled["old_label"] = df["pseudo_label"].values

    Path(p["output"]).parent.mkdir(parents=True, exist_ok=True)
    labeled[["text", "pseudo_label"]].to_csv(p["output"], index=False)

    changed = labeled["pseudo_label"] != labeled["old_label"]
    summary = {
        "changed": int(changed.sum()),
        "changed_fraction": round(float(changed.mean()), 4),
        "before": labeled["old_label"].value_counts().to_dict(),
        "after": labeled["pseudo_label"].value_counts().to_dict(),
    }
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(reports_dir) / "relabel.json").write_text(json.dumps(summary, indent=2))
    print(pd.crosstab(labeled["old_label"], labeled["pseudo_label"], rownames=["przed"], colnames=["po"]))
    return labeled


if __name__ == "__main__":
    relabel(load_params())
