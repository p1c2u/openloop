FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# git + CA certs are needed by the coding worker, which shells out to `git`
# to clone/commit/push over HTTPS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

COPY agents ./agents

EXPOSE 8000

CMD ["uvicorn", "openloop.app:app", "--host", "0.0.0.0", "--port", "8000"]
