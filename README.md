# Analiza sentymentu komentarzy o kryptowalutach modelem CryptoBERT

![CI](https://github.com/TWOJ_LOGIN/crypto-sentiment/actions/workflows/ci.yml/badge.svg)

Dostrajanie modelu [ElKulako/cryptobert](https://huggingface.co/ElKulako/cryptobert) na komentarzach z subreddita r/CryptoCurrency
(etykiety: głosowanie VADER, kk08/CryptoBERT i ElKulako/CryptoBERT) wraz z pełnym procesem MLOps:
Docker, CI (GitHub Actions), potok w Jenkinsie, DVC + MLflow (DagsHub) oraz monitoring API (Prometheus + Grafana).

## Architektura

```
            ┌──────────── GitHub ────────────┐        ┌──────────── DagsHub ────────────┐
 git push → │ GitHub Actions (CI)            │        │ DVC remote: dane i modele       │
            │  lint → testy → obrazy Docker  │◄──────►│ MLflow: eksperymenty, metryki,  │
            │  → szybki potok ML (500 próbek)│        │         Model Registry          │
            └────────────────────────────────┘        └─────────────────────────────────┘
                                                                    ▲
 Jenkins (komputer z GPU) ──────────────────────────────────────────┤
  dvc pull → parametry → dvc repro (prepare → train → evaluate)     │
  → próg jakości (F1) → rejestracja modelu → wdrożenie API ─────────┘
                                                   │
                                     API (FastAPI) /predict ──► Prometheus ──► Grafana
```

| Wymaganie | Realizacja |
|---|---|
| 1. Docker | `Dockerfile.train` (środowisko ML: JupyterLab, PyTorch CPU/CUDA, DVC, MLflow), `Dockerfile` (API), `docker-compose.yml` z profilami `train` / `serve` / `ci`, nakładka GPU `docker-compose.gpu.yml` |
| 2. CI | `.github/workflows/ci.yml`: ruff, walidacja potoku DVC, pytest (w tym trening na malutkim modelu i test API), budowa i publikacja obrazów w GHCR, szybki przebieg potoku z metrykami w podsumowaniu |
| 3. Jenkins | `Jenkinsfile`: parametryzowany potok w kontenerze z GPU – DVC → trening → ewaluacja → próg jakości → rejestracja w MLflow → wdrożenie API i test dymny |
| 4. Wizualizacja i monitoring | MLflow (krzywe uczenia, porównanie przebiegów, macierz pomyłek), `dvc plots`, wykresy w notebooku, monitoring API w Grafanie (ruch, czas odpowiedzi, rozkład predykcji, pewność modelu) |
| 5. DVC + MLflow | DVC: wersjonowanie danych i modelu, potok `dvc.yaml` + `params.yaml`; MLflow: śledzenie eksperymentów, poszukiwanie hiperparametrów, Model Registry |

## Struktura

```
src/
  dataset.py        budowa zbioru: Reddit → filtrowanie → pseudo-etykiety (VADER + slang krypto) → augmentacja
  fetch_reddit.py   pobranie nowych komentarzy z Reddita i dołączenie do zbioru (Jenkins: FETCH_REDDIT)
  relabel.py        etap relabel: ponowne etykietowanie zbioru ze słownikiem slangu krypto
  data.py           etap prepare: podział train / val / test
  train.py          etap train: dostrajanie + MLflow
  evaluate.py       etap evaluate: metryki, macierz pomyłek, błędne predykcje
  predict.py        predykcja (wspólna dla ewaluacji, notebooka i API)
  sweep.py          poszukiwanie hiperparametrów (Optuna + MLflow)
  register.py       rejestracja modelu w MLflow Model Registry
  check_metrics.py  próg jakości
app/main.py         API (FastAPI + metryki Prometheusa)
notebooks/          notebook – trening krok po kroku z użyciem src/
dvc.yaml            potok DVC, params.yaml – parametry
Jenkinsfile, jenkins/   potok Jenkinsa i obraz Jenkinsa
monitoring/         Prometheus i dashboard Grafany
tests/              testy (pytest)
```

## Konfiguracja (jednorazowo)

1. **DagsHub:** załóż konto na [dagshub.com](https://dagshub.com), utwórz repozytorium przez *Connect a repository → GitHub*
   (połączone z repozytorium na GitHubie) i wygeneruj token (*Settings → Tokens*).
2. **Plik `.env`:** `cp .env.example .env` i uzupełnij login, token i adres MLflow.
3. **DVC:** wpisz swój login w adresie zdalnego repozytorium i wyślij dane:
   ```bash
   dvc remote modify origin url https://dagshub.com/irintinuviel/crypto_sentiment.dvc
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user irintinuviel
   dvc remote modify origin --local password TWOJ_TOKEN
   dvc push
   ```
4. **GitHub:** w *Settings → Secrets and variables → Actions* dodaj sekrety `DAGSHUB_USER`, `DAGSHUB_TOKEN`
   oraz zmienną `DAGSHUB_REPO` (np. `irintinuviel/crypto_sentiment`).
5. Zamień `TWOJ_LOGIN` w `Jenkinsfile` i w tym README.

## 1. Trening – Jupyter w Dockerze

```bash
# GPU (GTX 1660 Ti, wymaga sterownika NVIDIA; w Windows: Docker Desktop + WSL2)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile train up --build
# CPU
docker compose --profile train up --build
```

JupyterLab: http://localhost:8888/notebooks/cryptobert.ipynb

Ten sam proces bez notebooka (w kontenerze lub lokalnie):

```bash
dvc pull                  # dane (i ostatni model)
dvc repro                 # relabel → prepare → train → evaluate (tylko zmienione etapy)
dvc metrics show          # metryki
dvc plots show            # wykresy → dvc_plots/index.html
dvc params diff / dvc metrics diff   # porównanie z poprzednim commitem
python -m src.sweep --trials 9 --apply   # poszukiwanie hiperparametrów
git add dvc.lock params.yaml && git commit -m "..." && dvc push   # zapis nowej wersji modelu
```

Eksperymenty, krzywe uczenia i zarejestrowane modele: zakładka *Experiments* w repozytorium na DagsHub (MLflow UI).

## 2. CI – GitHub Actions

Uruchamiane przy każdym `push` i pull requeście:

1. `ruff check` i `dvc dag` (poprawność potoku),
2. `pytest`: testy przetwarzania danych oraz test dymny prepare → train → evaluate → API na malutkim modelu,
3. budowa obrazów `api` i `train` (na `main` publikowane w `ghcr.io`),
4. na `main`: `dvc pull` z DagsHub i potok na 500 próbkach (1 epoka, CPU) – metryki w podsumowaniu przebiegu,
   przebieg w eksperymencie MLflow `ci`, raporty jako artefakt.

## 3. Jenkins (komputer z GPU)

```bash
docker compose --profile ci up -d --build
```

1. http://localhost:8080 – hasło startowe: `docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword`.
2. *Manage Jenkins → Credentials*: dodaj *Username with password* o ID `dagshub` (login i token DagsHub).
   Opcjonalnie ID `reddit` (client id i secret Reddit API) – potrzebne tylko przy parametrze `FETCH_REDDIT`.
3. *New Item → Pipeline → Pipeline script from SCM* → Git, adres repozytorium, ścieżka `Jenkinsfile`.
4. *Build with Parameters* – epoki, learning rate, batch, rozmiar próbki, próg F1, GPU, rejestracja, wdrożenie.

Etapy: **Dane (DVC) → Parametry → Trening i ewaluacja (`dvc repro`, GPU) → Próg jakości → Rejestracja modelu
(MLflow) → Wdrożenie API**. Raporty (metryki, macierz pomyłek) są archiwizowane przy każdym buildzie.

## 4. API i monitoring

```bash
docker compose --profile serve up --build
```

| Usługa | Adres |
|---|---|
| API | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana dashboard | http://localhost:3000 (admin / admin) |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["bitcoin is going to the moon", "eth fees are killing me"]}'
```

```json
[
  {"text": "bitcoin is going to the moon", "label": "positive",
   "scores": {"negative": 0.01, "neutral": 0.04, "positive": 0.95}},
  ...
]
```

| Endpoint | Metoda | Opis |
|---|---|---|
| `/predict` | POST | `{"texts": [...]}` (1–64 tekstów) → etykieta i prawdopodobieństwa |
| `/health` | GET | stan API |
| `/metrics` | GET | metryki Prometheusa |

## Testy lokalnie

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
ruff check . && pytest -v
```
