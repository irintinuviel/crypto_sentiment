"""Pobranie nowych komentarzy z Reddita i dołączenie ich do zbioru (wywoływane z Jenkinsa: FETCH_REDDIT).

Nowe teksty przechodzą te same kroki co przy budowie zbioru: filtrowanie, pseudo-etykiety, augmentacja.
Po uruchomieniu nowa wersja danych trafia do DVC: dvc add data/final_crypto_data.csv && dvc push

    python -m src.fetch_reddit --limit 100
"""
import argparse
from pathlib import Path

import pandas as pd

from src.dataset import augment, filter_comments, pseudo_label, scrape_reddit

DATASET = "data/final_crypto_data.csv"


def fetch_reddit(limit: int, subreddit: str = "CryptoCurrency", path: str = DATASET) -> pd.DataFrame:
    existing = pd.read_csv(path) if Path(path).exists() else pd.DataFrame(columns=["text", "pseudo_label"])

    clean = filter_comments(scrape_reddit(subreddit, limit=limit))
    new = clean[~clean["text"].isin(existing["text"])]
    print(f"Reddit: {len(clean)} komentarzy po filtrowaniu, nowych: {len(new)}")
    if new.empty:
        return existing

    labeled = augment(pseudo_label(new), n=3)
    result = pd.concat([existing, labeled], ignore_index=True).drop_duplicates(subset="text")
    result.to_csv(path, index=False)
    print(f"Zbiór: {len(existing)} -> {len(result)} tekstów")
    print(labeled["pseudo_label"].value_counts().to_string())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100,
                        help="liczba najpopularniejszych postów z ostatniego roku")
    parser.add_argument("--subreddit", default="CryptoCurrency")
    args = parser.parse_args()
    fetch_reddit(args.limit, args.subreddit)
