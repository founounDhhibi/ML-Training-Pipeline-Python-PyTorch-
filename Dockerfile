FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY README.md ./README.md

RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/artifacts /app/logs \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "src.main", "--epochs", "1", "--batch-size", "64", "--num-workers", "0"]
