FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction

COPY src/ src/
COPY models/ models/
COPY examples/ examples/

RUN useradd -m appuser
USER appuser

EXPOSE 7860

CMD ["python", "-m", "adma.app"]
