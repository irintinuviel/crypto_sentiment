# Analiza sentymentu komentarzy o kryptowalutach (CryptoBERT)

Dostrajanie modelu [ElKulako/cryptobert](https://huggingface.co/ElKulako/cryptobert) na komentarzach z r/CryptoCurrency
z etykietami nadanymi przez głosowanie trzech modeli (VADER, kk08/CryptoBERT, ElKulako/CryptoBERT)
oraz udostępnienie najlepszego modelu jako REST API w Dockerze.

## Struktura

- `fine-tuning.ipynb` – pobranie danych, czyszczenie, etykietowanie, augmentacja, dostrajanie, ewaluacja i eksport modelu
- `final_crypto_data.csv` – gotowy zbiór danych (`text`, `pseudo_label`)
- `app/main.py` – API (FastAPI) zwracające predykcje najlepszego modelu
- `model/` – miejsce na wyeksportowany model (poza gitem)

## 1. Trening (Jupyter)

```bash
pip install torch
pip install -r requirements-train.txt
jupyter lab
```

Uruchom komórki dostrajania, a następnie ostatnią komórkę **Eksport najlepszego modelu do API** –
zapisze ona model do folderu `model/`.

## 2. API w Dockerze

```bash
docker compose up --build
```

API działa pod adresem http://localhost:8000, interaktywna dokumentacja: http://localhost:8000/docs.

Model z Hugging Face Hub zamiast lokalnego folderu:

```bash
docker build -t crypto-sentiment-api .
docker run -p 8000:8000 -e MODEL_PATH=TWOJ_LOGIN/cryptobert-sentiment crypto-sentiment-api
```

### Przykład zapytania

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["bitcoin is going to the moon", "eth fees are killing me"]}'
```

Odpowiedź:

```json
[
  {"text": "bitcoin is going to the moon", "label": "positive",
   "scores": {"negative": 0.01, "neutral": 0.04, "positive": 0.95}},
  ...
]
```

| Endpoint | Metoda | Opis |
|---|---|---|
| `/health` | GET | stan API i używany model |
| `/predict` | POST | `{"texts": [...]}` (1–64 tekstów) → etykieta i prawdopodobieństwa |

## Uwagi

- Komórka pobierająca dane wymaga własnych kluczy Reddit API (`client_id`, `client_secret`).
- Etap etykietowania i augmentacji można pominąć – wynik jest już w `final_crypto_data.csv`.
- Obraz używa PyTorcha w wersji CPU – do predykcji w zupełności wystarcza.
