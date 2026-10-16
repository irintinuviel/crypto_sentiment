import re

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))


def clean_text(text: str) -> str:
    """Usunięcie linków i konwersja do małych liter (tak jak przy budowie zbioru)."""
    return re.sub(r"http\S+", "", str(text)).lower()
