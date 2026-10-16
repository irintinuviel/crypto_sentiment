"""Budowa zbioru danych: pobranie komentarzy z Reddita, filtrowanie, pseudo-etykietowanie i augmentacja.

Wynik (data/final_crypto_data.csv) jest wersjonowany w DVC, więc te kroki uruchamia się tylko przy budowie
nowej wersji zbioru (wymagają kluczy Reddit API i długo trwają).
"""
import os
import re

import pandas as pd

from src.text import clean_text

CRYPTO_NAMES = ["bitcoin", "btc", "eth", "ethereum", "sol", "solana", "xrp", "ripple",
                "bnb", "binance", "ada", "cardano", "bch", "usdt", "tether",
                "monero", "shib", "doge", "litecoin", "ltc"]
CRYPTO_PATTERN = r"\b(" + "|".join(re.escape(word) for word in CRYPTO_NAMES) + r")\b"
TELEGRAM_MOD_MESSAGE = "Your comment contained a link to telegram, which is hard blocked by reddit."


def scrape_reddit(subreddit_name: str = "CryptoCurrency", limit: int = 1000) -> pd.DataFrame:
    """Pobranie tytułów i komentarzy z najpopularniejszych postów (klucze w REDDIT_CLIENT_ID/SECRET)."""
    import praw

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.getenv("REDDIT_USER_AGENT", "crypto-sentiment"),
    )
    data = []
    for submission in reddit.subreddit(subreddit_name).top(time_filter="year", limit=limit):
        data.append(submission.title)
        submission.comments.replace_more(limit=0)
        data.extend(comment.body for comment in submission.comments.list())
    return pd.DataFrame({"text": data})


def filter_comments(df: pd.DataFrame, min_words: int = 4, max_words: int = 100) -> pd.DataFrame:
    """Czyszczenie i wybór komentarzy o długości 4–100 słów dotyczących dokładnie jednej kryptowaluty."""
    df = df.dropna().drop_duplicates()
    df = df[~df["text"].str.contains(TELEGRAM_MOD_MESSAGE, regex=False)].copy()
    df["text"] = df["text"].apply(clean_text)

    words = df["text"].str.split().str.len()
    df = df[(words >= min_words) & (words <= max_words)]
    matches = df["text"].apply(lambda x: len(re.findall(CRYPTO_PATTERN, x)))
    return df[matches == 1].reset_index(drop=True)


def pseudo_label(df: pd.DataFrame) -> pd.DataFrame:
    """Etykietowanie głosowaniem większościowym: VADER, kk08/CryptoBERT, ElKulako/CryptoBERT."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    vader = SentimentIntensityAnalyzer()
    models = {
        name: (AutoTokenizer.from_pretrained(name),
               AutoModelForSequenceClassification.from_pretrained(name).eval(), labels)
        for name, labels in [("kk08/CryptoBERT", ["negative", "positive"]),
                             ("ElKulako/cryptobert", ["negative", "neutral", "positive"])]
    }

    @torch.no_grad()
    def classify(text: str) -> tuple[str, list[str]]:
        score = vader.polarity_scores(text)["compound"]
        votes = ["positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"]
        for tokenizer, model, labels in models.values():
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            votes.append(labels[int(model(**tokens).logits.argmax(dim=1))])
        final = "neutral" if len(set(votes)) == 3 else max(set(votes), key=votes.count)
        return final, votes

    df = df.copy()
    df["pseudo_label"], df["model_votes"] = zip(*df["text"].apply(classify))
    return df


def augment(df: pd.DataFrame, n: int = 3, labels: tuple[str, ...] = ("positive", "negative"),
            seed: int = 42) -> pd.DataFrame:
    """Augmentacja synonimami (WordNet) klas mniejszościowych – każdy tekst dostaje n wariantów."""
    import nlpaug.augmenter.word as naw
    import nltk

    for resource in ("wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"):
        nltk.download(resource, quiet=True)
    augmenter = naw.SynonymAug(aug_src="wordnet", aug_max=3)

    df = df[["text", "pseudo_label"]]
    augmented = []
    for label in labels:
        for text in df.loc[df["pseudo_label"] == label, "text"]:
            try:
                augmented += [{"text": s, "pseudo_label": label} for s in augmenter.augment(text, n=n)]
            except Exception as e:
                print(f"błąd: {text[:30]}... ({e})")

    result = pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)
