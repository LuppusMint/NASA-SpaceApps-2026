FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir duckdb pyarrow numpy pandas

EXPOSE 8080

CMD ["sh", "-c", "python -m app.server --processed-dir data/processed --host 0.0.0.0 --port ${PORT:-8080}"]
