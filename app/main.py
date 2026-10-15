import os
import re
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# lokalny folder z modelem albo nazwa repozytorium na Hugging Face Hub
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))

# kolejność etykiet taka jak podczas dostrajania (label_map w notebooku)
LABELS = ["negative", "neutral", "positive"]

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH)
    ml["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    ml["model"].eval()
    yield
    ml.clear()


app = FastAPI(title="CryptoBERT – analiza sentymentu", lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64,
                             examples=[["bitcoin is going to the moon", "eth fees are killing me"]])


class Prediction(BaseModel):
    text: str
    label: str
    scores: dict[str, float]


def clean_text(text: str) -> str:
    # to samo czyszczenie co przy przygotowaniu danych treningowych
    return re.sub(r"http\S+", "", text).lower()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=list[Prediction])
def predict(request: PredictRequest):
    tokenizer, model = ml["tokenizer"], ml["model"]
    inputs = tokenizer([clean_text(t) for t in request.texts], return_tensors="pt",
                       padding=True, truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)

    return [
        Prediction(
            text=text,
            label=LABELS[int(p.argmax())],
            scores={label: round(float(score), 4) for label, score in zip(LABELS, p)},
        )
        for text, p in zip(request.texts, probs)
    ]
