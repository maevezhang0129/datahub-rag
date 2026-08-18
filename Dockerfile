FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DRR_MODEL_CACHE=/models

WORKDIR /app

# No apt layer: psycopg[binary], torch and tokenizers all ship prebuilt wheels,
# so the image needs no compiler and stays small.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Bake the embedding model into the image so `docker compose up` does not need
# to reach Hugging Face on first run.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='/models')"

COPY . .

CMD ["uvicorn", "drr_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
