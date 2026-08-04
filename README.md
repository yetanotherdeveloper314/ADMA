# Autonomous Detection of Military Assets

**ADMA (Autonomous Detection of Military Assets)** detects and classifies military assets — tanks, fighter jets, drones, helicopters, warships, missiles, and more — in any image using YOLOv8 object detection and a [Gradio](https://www.gradio.app/) web interface.

---

## Prerequisites

- **Python 3.12+**
- **Poetry** — install via `pip install poetry` or [the official installer](https://python-poetry.org/docs/#installation)
- **Roboflow API key** (free) — needed only for downloading training datasets
  1. Sign up at [app.roboflow.com](https://app.roboflow.com/)
  2. Go to **Settings → API Keys**
  3. Copy your **Private API Key**

> **Note:** No pre-trained model weights are included in the repo. You must train a model (see below) or provide your own `best.pt` file at `models/<model_name>/best.pt` before the web app will work.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YourUsername/ADMA.git
cd ADMA
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your Roboflow API key:

```
ROBOFLOW_API_KEY="your_key_here"
```

### 3. Install dependencies

```bash
# Main dependencies only
poetry install

# Include dev dependencies (needed for dataset downloading)
poetry install --with dev
```

### 4. Download datasets

```bash
poetry run python scripts/download_dataset.py --dataset all
```

### 5. Train a model

```bash
# Train on all datasets combined (recommended)
poetry run python scripts/train.py --dataset all --epochs 150 --model-size m

# Or train on a single dataset
poetry run python scripts/train.py --dataset military_vehicles --epochs 150 --model-size m
```

**Model sizes:** `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (extra-large)
Larger models are more accurate but slower and require more GPU memory.

> **GPU recommended.** Training on CPU is very slow. Use `--model-size n` and fewer `--epochs` for CPU-only training, or use [Google Colab](https://colab.research.google.com/) with the included `deploy_colab.ipynb` notebook for free GPU access.

### 6. Launch the web app

```bash
poetry run python -m adma.app
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

### 7. Test on a single image (CLI)

```bash
poetry run python scripts/test_detection.py --image path/to/photo.jpg --model all_m

# List all available trained models
poetry run python scripts/test_detection.py --list-models
```

---

## Alternative: Run with Docker

Make sure you have a trained model in `models/` before building.

```bash
docker compose up --build
```

Open [http://localhost:7860](http://localhost:7860).

---

## Alternative: Run with pip (no Poetry)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install ultralytics>=8.0.0 gradio>=4.0.0 Pillow>=10.0.0 numpy>=1.24.0 \
            opencv-python-headless>=4.8.0 pyyaml>=6.0 python-dotenv>=1.0.0

# For downloading datasets, also install roboflow
pip install roboflow>=1.1.0

# Launch the app (after training / providing a model)
python -m adma.app
```

---

## Project Structure

```
ADMA/
├── src/adma/              Python package
│   ├── app.py             Gradio web application (entry point)
│   ├── config.py          Configuration from environment variables
│   ├── detector.py        YOLOv8 inference engine
│   └── datasets.py        Dataset registry & label normalization
├── scripts/               CLI tools
│   ├── download_dataset.py   Download datasets from Roboflow
│   ├── train.py              Fine-tune YOLOv8 models
│   └── test_detection.py     Run detection on a single image
├── pyproject.toml         Poetry dependencies & scripts
├── Dockerfile             Container build
├── docker-compose.yml     One-command container deployment
├── deploy_colab.ipynb     Google Colab notebook
└── .env.example           Environment variable template
```

**Runtime directories (created as needed, gitignored):**

| Directory  | Purpose                                |
| ---------- | -------------------------------------- |
| `models/`  | Trained model weights (`best.pt`)      |
| `data/`    | Downloaded Roboflow datasets           |
| `runs/`    | Ultralytics training artifacts & plots |
| `output/`  | Annotated images from CLI testing      |

---

## Environment Variables

All variables have sensible defaults. See `.env.example` for the full list.

| Variable              | Default        | Description                        |
| --------------------- | -------------- | ---------------------------------- |
| `ROBOFLOW_API_KEY`    | *(none)*       | Required for dataset download      |
| `ADMA_DATA_DIR`       | `data`         | Dataset storage path               |
| `ADMA_MODELS_DIR`     | `models`       | Trained model weights path         |
| `ADMA_DEFAULT_MODEL`  | `tank_images_m`| Model loaded on app startup        |
| `ADMA_TRAIN_EPOCHS`   | `100`          | Training epochs                    |
| `ADMA_TRAIN_BATCH_SIZE`| `32`          | Training batch size                |
| `ADMA_SERVER_PORT`    | `7860`         | Web app port                       |

---

## Datasets

Four datasets are registered and downloaded from [Roboflow](https://roboflow.com/):

| Key                    | Description                     | Images |
| ---------------------- | ------------------------------- | ------ |
| `tank_images`          | Tank Images                     | ~6.7K  |
| `military_vehicles`    | Military Vehicle Detection      | ~3.1K  |
| `military_vehicles_obj`| Military Vehicles Obj Detection | ~2.1K  |
| `military_objects`     | Military Object Detection       | ~2.3K  |

When training on `--dataset all`, labels are automatically normalized across datasets into a unified set of military classes:
`tank`, `military_vehicle`, `drone`, `fighter_jet`, `armored_vehicle`, `military_aircraft`, `military_helicopter`, `stealth_aircraft`, `missile`, `missile_launcher`, `military_truck`, `warship`, `weapons`
