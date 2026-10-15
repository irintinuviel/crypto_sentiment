FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

# PyTorch w wersji CPU (dużo mniejszy obraz niż wersja z CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# zasoby NLTK potrzebne do augmentacji
RUN python -m nltk.downloader -d $NLTK_DATA wordnet omw-1.4 averaged_perceptron_tagger_eng

COPY fine-tuning.ipynb final_crypto_data.csv ./

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''"]
