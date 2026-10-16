import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.predict import Predictor

PREDICTIONS = Counter("sentiment_predictions_total", "Liczba predykcji według etykiety", ["label"])
CONFIDENCE = Histogram("sentiment_confidence", "Pewność modelu dla przewidzianej etykiety",
                       buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # lokalny folder z modelem albo nazwa repozytorium na Hugging Face Hub
    model_path = os.getenv("MODEL_PATH", "/app/model")
    ml["predictor"] = Predictor(model_path, max_length=int(os.getenv("MAX_LENGTH", "128")))
    ml["model_path"] = model_path
    yield
    ml.clear()


app = FastAPI(title="CryptoBERT – analiza sentymentu", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, include_in_schema=False)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64,
                             examples=[["bitcoin is going to the moon", "eth fees are killing me"]])


class Prediction(BaseModel):
    text: str
    label: str
    scores: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model": ml.get("model_path")}


@app.post("/predict", response_model=list[Prediction])
def predict(request: PredictRequest):
    results = ml["predictor"].predict(request.texts)
    for result in results:
        PREDICTIONS.labels(result["label"]).inc()
        CONFIDENCE.observe(result["scores"][result["label"]])
    return [Prediction(text=text, **result) for text, result in zip(request.texts, results)]
