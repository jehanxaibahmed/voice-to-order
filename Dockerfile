FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY samples/catalog.json ./samples/catalog.json

RUN useradd --create-home app && mkdir -p /app/data && chown app /app/data
USER app

ENV VTO_WORK_DIR=/app/data
EXPOSE 8000
CMD ["uvicorn", "voice_to_order.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
