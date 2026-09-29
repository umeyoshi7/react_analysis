# Vertex AI カスタムジョブ用の実行イメージ (ビルドの文脈は dft/ ディレクトリ)
#   gcloud builds submit dft --tag asia-northeast1-docker.pkg.dev/<PROJECT>/hf-dft/hf-dft:latest
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hf_dft/ hf_dft/
COPY reference_set.csv .

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "hf_dft"]
