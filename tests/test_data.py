import pandas as pd
import pytest

from src.data import load_dataset, prepare, split_dataset
from src.dataset import filter_comments
from src.params import override
from src.text import clean_text


def test_clean_text_removes_links_and_lowercases():
    assert clean_text("BTC up! https://x.com/abc now") == "btc up!  now"


def test_load_dataset_maps_labels_and_drops_invalid(dataset_csv):
    df = load_dataset(dataset_csv)
    assert len(df) == 120
    assert set(df["label"]) == {0, 1, 2}


def test_split_is_stratified_and_disjoint(dataset_csv):
    df = load_dataset(dataset_csv)
    splits = split_dataset(df, val_size=0.1, test_size=0.1, seed=42)
    assert sum(len(s) for s in splits.values()) == len(df)
    assert set(splits["train"].index).isdisjoint(splits["test"].index)
    for split in splits.values():
        assert split["label"].value_counts().nunique() == 1  # równe proporcje klas


def test_sample_size_limits_dataset(dataset_csv):
    splits = split_dataset(load_dataset(dataset_csv), 0.1, 0.1, seed=42, sample_size=60)
    assert sum(len(s) for s in splits.values()) == 60


def test_prepare_writes_splits(params, tmp_path):
    prepare(params, out_dir=tmp_path / "processed")
    for name in ("train", "val", "test"):
        assert (tmp_path / "processed" / f"{name}.csv").exists()


def test_filter_comments_keeps_single_crypto_mentions():
    df = pd.DataFrame({"text": [
        "Bitcoin is going up again today",           # OK
        "bitcoin and eth are both pumping hard",     # dwie kryptowaluty
        "btc moon",                                  # za krótki
        "I love this coin so much right now",        # brak kryptowaluty
    ]})
    assert filter_comments(df)["text"].tolist() == ["bitcoin is going up again today"]


def test_override_params():
    params = {"train": {"epochs": 5, "learning_rate": 2e-5}}
    updated = override(params, ["train.epochs=1", "train.learning_rate=1e-4"])
    assert updated["train"] == {"epochs": 1, "learning_rate": 1e-4}
    assert params["train"]["epochs"] == 5
    with pytest.raises(KeyError):
        override(params, ["train.unknown=1"])
