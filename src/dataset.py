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


CRYPTO_LEXICON = {
    # pozytywne
    "moon": 2.5, "mooning": 2.5, "moonshot": 2.0, "rocket": 1.5, "bullish": 2.5, "bull": 1.5,
    "pump": 1.5, "pumping": 1.5, "hodl": 1.5, "hodling": 1.5, "lambo": 2.0, "ath": 2.0,
    "gains": 2.0, "rally": 2.0, "rallying": 2.0, "surge": 2.0, "surging": 2.0, "skyrocket": 2.5,
    "skyrocketing": 2.5, "soaring": 2.0, "breakout": 1.5, "undervalued": 1.5, "wagmi": 2.0,
    "goingup": 1.5,
    # negatywne
    "bearish": -2.5, "bear": -1.5, "dump": -2.0, "dumping": -2.0, "dumped": -2.0, "rekt": -2.5,
    "fud": -1.5, "scam": -3.0, "scammer": -3.0, "rugpull": -3.0, "rugged": -3.0, "ponzi": -3.0,
    "bagholder": -2.0, "bagholders": -2.0, "overvalued": -1.5, "plummet": -2.5, "plummeting": -2.5,
    "tanking": -2.5, "tanked": -2.5, "liquidated": -2.5, "ngmi": -2.0, "capitulation": -2.0,
    "goingdown": -1.5,
}

CRYPTO_PHRASES = {
    "to the moon": "moon", "going up": "goingup", "goes up": "goingup", "went up": "goingup",
    "going down": "goingdown", "goes down": "goingdown", "went down": "goingdown",
    "rug pull": "rugpull", "bag holder": "bagholder", "all time high": "ath",
}

LABELER_MODELS = [("kk08/CryptoBERT", ["negative", "positive"]),
                  ("ElKulako/cryptobert", ["negative", "neutral", "positive"])]


def crypto_vader():
    """VADER ze słownikiem rozszerzonym o slang kryptowalutowy."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(CRYPTO_LEXICON)
    return vader


def vader_vote(vader, text: str) -> str:
    text = str(text).lower()
    for phrase, token in CRYPTO_PHRASES.items():
        text = text.replace(phrase, token)
    score = vader.polarity_scores(text)["compound"]
    return "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"


def model_votes(texts: list[str], batch_size: int = 64) -> list[list[str]]:
    """Głosy modeli kk08/CryptoBERT i ElKulako/CryptoBERT (wsadowo, na GPU jeśli jest dostępne)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    votes = [[] for _ in texts]
    for name, labels in LABELER_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = [str(t) for t in texts[start:start + batch_size]]
                tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                   max_length=512).to(device)
                for i, pred in enumerate(model(**tokens).logits.argmax(dim=1).tolist()):
                    votes[start + i].append(labels[pred])
        print(f"{name}: {len(texts)} tekstów")
    return votes


def majority(votes: list[str]) -> str:
    """Głosowanie większościowe; trzy różne głosy oznaczają brak zgody, czyli neutral."""
    return "neutral" if len(set(votes)) == 3 else max(set(votes), key=votes.count)


def pseudo_label(df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """Głosowanie większościowe: VADER (+ slang krypto), kk08/CryptoBERT, ElKulako/CryptoBERT."""
    vader = crypto_vader()
    texts = df["text"].tolist()
    votes = [[vader_vote(vader, text)] + other for text, other in zip(texts, model_votes(texts, batch_size))]

    df = df.copy()
    df["pseudo_label"] = [majority(v) for v in votes]
    df["model_votes"] = votes
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
