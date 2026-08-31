"""
Central configuration for ADMA (Autonomous Detection of Military Assets).

All settings load from environment variables with sensible defaults for
non-sensitive values.  Create a ``.env`` file in the project root
(copy ``.env.example``) to override defaults without touching code.

Sensitive values (like API keys) have NO hardcoded fallback and MUST
be set via environment variable or ``.env``.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise OSError(
            f"Required environment variable '{key}' is not set.\n"
            f"Add it to your .env file or export it in your shell.\n"
            f"See .env.example for reference."
        )
    return val


def _env_or_none(key: str) -> str | None:
    return os.environ.get(key)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


# API Keys
ROBOFLOW_API_KEY: str | None = _env_or_none("ROBOFLOW_API_KEY")

# Paths
DATA_DIR: str = _env("ADMA_DATA_DIR", "data")
MODELS_DIR: str = _env("ADMA_MODELS_DIR", "models")
RUNS_DIR: str = _env("ADMA_RUNS_DIR", "runs")
OUTPUT_DIR: str = _env("ADMA_OUTPUT_DIR", "output")
EXAMPLES_DIR: str = _env("ADMA_EXAMPLES_DIR", "examples")
COMBINED_DATA_DIR: str = _env("ADMA_COMBINED_DATA_DIR", "data/combined")

# Model Defaults
DEFAULT_MODEL: str = _env("ADMA_DEFAULT_MODEL", "tank_images_m")
DEFAULT_MODEL_SIZE: str = _env("ADMA_DEFAULT_MODEL_SIZE", "m")
DEFAULT_CONFIDENCE: float = _env_float("ADMA_DEFAULT_CONFIDENCE", 0.25)

# Training
TRAIN_EPOCHS: int = _env_int("ADMA_TRAIN_EPOCHS", 100)
TRAIN_BATCH_SIZE: int = _env_int("ADMA_TRAIN_BATCH_SIZE", 16)
TRAIN_IMAGE_SIZE: int = _env_int("ADMA_TRAIN_IMAGE_SIZE", 640)
TRAIN_WORKERS: int = _env_int("ADMA_TRAIN_WORKERS", 8)
TRAIN_PATIENCE: int = _env_int("ADMA_TRAIN_PATIENCE", 15)

# Server
SERVER_HOST: str = _env("ADMA_SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = _env_int("ADMA_SERVER_PORT", 7860)
SERVER_SHARE: bool = _env_bool("ADMA_SERVER_SHARE", False)

# Data Processing
VAL_SPLIT_RATIO: float = _env_float("ADMA_VAL_SPLIT_RATIO", 0.2)
RANDOM_SEED: int = _env_int("ADMA_RANDOM_SEED", 42)
