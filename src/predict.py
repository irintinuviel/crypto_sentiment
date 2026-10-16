import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.text import LABELS, clean_text


class Predictor:
    """Predykcja sentymentu – wspólna dla ewaluacji, notebooka i API."""

    def __init__(self, model_path: str, max_length: int = 128, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 64) -> list[dict]:
        results = []
        for start in range(0, len(texts), batch_size):
            batch = [clean_text(t) for t in texts[start:start + batch_size]]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.max_length).to(self.device)
            probs = torch.softmax(self.model(**inputs).logits, dim=1).cpu()
            for p in probs:
                results.append({
                    "label": LABELS[int(p.argmax())],
                    "scores": {label: round(float(s), 4) for label, s in zip(LABELS, p)},
                })
        return results
