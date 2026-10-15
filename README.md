# Analiza sentymentu komentarzy o kryptowalutach (CryptoBERT)

Dostrajanie modelu [ElKulako/cryptobert](https://huggingface.co/ElKulako/cryptobert) na komentarzach z r/CryptoCurrency
z etykietami nadanymi przez głosowanie trzech modeli (VADER, kk08/CryptoBERT, ElKulako/CryptoBERT).

## Zawartość

- `fine-tuning.ipynb` – pobranie danych, czyszczenie, etykietowanie, augmentacja, dostrajanie, ewaluacja
- `final_crypto_data.csv` – gotowy zbiór danych (`text`, `pseudo_label`)

## Uruchomienie w Dockerze

```bash
cp .env.example .env        # uzupełnij WANDB_API_KEY
docker compose up --build
```

Następnie otwórz http://localhost:8888.

Bez docker compose:

```bash
docker build -t crypto-sentiment .
docker run -p 8888:8888 -e WANDB_API_KEY=... -v "$(pwd)":/app crypto-sentiment
```

Obraz zawiera PyTorch w wersji CPU – trenowanie działa, ale jest wolne. Do treningu na GPU
zalecany jest Google Colab lub obraz bazowy `pytorch/pytorch` z `--gpus all`.

## Uwagi

- Komórka pobierająca dane wymaga własnych kluczy Reddit API (`client_id`, `client_secret`).
- Etap etykietowania i augmentacji można pominąć – wynik jest już w `final_crypto_data.csv`.
