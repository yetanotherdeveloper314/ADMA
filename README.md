# Autonomous Detection of Military Assets

**ADMA (Autonomous Detection of Military Assets)** detects and classifies military assets such as tanks, armored cars, etc. in any image using YOLOv8 object detection.

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

# Dev dependencies: tests, Ruff, duplicate detection and MLflow (needed for training)
poetry install --with dev

# Optional groups
poetry install --with dataset-download   # Roboflow, for step 4
poetry install --with vlm                # Ollama and transformers, for zero-shot models
```

### 4. Download datasets

```bash
poetry run python scripts/download_dataset.py --dataset all
```

### 5. Build the clean train / val / test split

All training and evaluation use `data/combined_v1/`, built from the four downloaded datasets in two steps:

```bash
# Group copies and near-copies of the same photo (Roboflow file names and perceptual hashes)
poetry run python scripts/find_duplicates.py --threshold 5

# Assign each duplicate group to one split (70/15/15 of the usable images) and lock the test set
poetry run python scripts/make_splits.py
```

- `find_duplicates.py` writes `data/splits/duplicate_groups.csv` and contact sheets of sample groups in `data/splits/duplicate_review/` for checking by eye.
- `make_splits.py` applies `LABEL_MAP` from `src/adma/datasets.py` and skips the datasets in `COMBINED_DATASET_EXCLUSIONS`. Images with missing, unmapped or only dropped labels are **not copied**; they are listed in `data/splits/make_splits_label_review.csv` for review.
- It writes `data/combined_v1/data.yaml`, `split_manifest.csv`, and the locked test list `test_images.txt` with its fingerprint `test_images.sha256`.
- Running it again never overwrites the output. `--force-rebuild` rebuilds it, and changing the test list also needs `--allow-test-list-change` (only when a test-set reset has been agreed).

**Never train, tune or pick models on the test split.** Use val for all decisions; test is scored once, at the end of the project.

### 6. Train a model

```bash
poetry run python scripts/train.py --data data/combined_v1/data.yaml --model-size m --epochs 100 --name baseline_v2
```

Training stops early when the val score stops improving for 15 epochs, saves the model to `models/<name>/best.pt`, then scores it on val and attaches the report to the MLflow run (see [Experiment tracking](#experiment-tracking)).

> **Legacy:** `train.py --dataset all` builds the old `data/combined/` from Roboflow's original splits. Those splits share near-duplicate photos between train and val, have no test set, and keep images whose dropped boxes were deleted. Don't use it for project results, and don't compare its scores with `combined_v1` scores.

**Model sizes:** `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (extra-large)
Larger models are more accurate but slower and require more GPU memory.

> **GPU recommended.** Training on CPU is very slow. Use `--model-size n` and fewer `--epochs` for CPU-only training, or use [Google Colab](https://colab.research.google.com/) with the included `deploy_colab.ipynb` notebook for free GPU access.

### 7. Evaluate a model

```bash
# Run a trained model on the val split
poetry run python scripts/evaluator.py --data data/combined_v1/data.yaml --split val --model models/baseline_v2/best.pt

# Or score saved predictions from any model against any data.yaml
poetry run python scripts/evaluator.py --data path/to/data.yaml --pred-labels path/to/labels
poetry run python scripts/evaluator.py --data path/to/data.yaml --pred-json predictions.json
```

The evaluator reports COCO mAP50-95, mAP50 and mAP75, per-class precision / recall / F1 at an operating confidence (`--conf`, default `ADMA_DEFAULT_CONFIDENCE`), the confidence with the best mean F1, a confusion matrix, and per-image latency when it runs the model itself. Reports are written to `runs/eval/<name>/` (`metrics.json`, `per_class.csv`, `confusion_matrix.csv`).

Prediction sources:

- `--model` — a model name from `models/` or a path to any Ultralytics weights file.
- `--pred-labels` — YOLO txt files with a confidence column (`class xc yc w h conf`, e.g. Ultralytics `save_txt=True save_conf=True`). Class ids must follow the `data.yaml` order.
- `--pred-json` — detection records in pixel coordinates, in the same shape `MilitaryAssetDetector.detect()` returns plus an `image` key:
  `[{"image": "tank_images/img1.jpg", "class": "tank", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}]`

If a model's class names differ from the dataset's, pass `--class-map map.yaml` mapping model names to dataset names (`null` drops a class). Other models can also be evaluated from Python by passing any `image_path -> detections` callable to `adma.evaluator.predict_with`.

When image ids start with a source folder (as in `combined_v1`), reports also break scores down per source dataset and per object size (COCO small / medium / large).

#### Zero-shot models (no training)

These write `--pred-json` files for the evaluator. Install the extra libraries with `poetry install --with vlm`.

```bash
# Open-vocabulary detectors: class names as text prompts, with confidence scores
poetry run python scripts/open_vocab_predict.py --framework yolo-world --data data/combined_v1/data.yaml --out runs/open_vocab/yw_val.json
poetry run python scripts/open_vocab_predict.py --framework owlv2 --data data/combined_v1/data.yaml --out runs/open_vocab/owl_val.json

# Qwen2.5-VL through Ollama (run `ollama pull qwen2.5vl:7b` first); no confidence scores
poetry run python scripts/vlm_predict.py --data data/combined_v1/data.yaml --out runs/qwen/qwen_300.json --limit 300
```

Both scripts take `--draw N` to save the first N images with true boxes (green) and predicted boxes (red), and write `<out>.latency.json` with the speed per image.

### Experiment tracking

Every training run is recorded with [MLflow](https://mlflow.org/): settings, training curves, the data path, seed and git commit, and the evaluator's `metrics.json`. Turn on Ultralytics' MLflow logging once per machine (training stops with an error if it is off):

```bash
poetry run yolo settings mlflow=True
```

Browse runs with:

```bash
poetry run mlflow ui
```

and open the address it prints. Commit your changes before a real training run, so the recorded commit matches the code. Run data is stored locally in `mlruns/` (gitignored).

### 8. Launch the web app

```bash
poetry run python -m adma.app
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

### 9. Test on a single image (CLI)

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
│   ├── evaluator.py       Model-agnostic detection metrics
│   └── datasets.py        Dataset registry & label normalization
├── scripts/               CLI tools
│   ├── download_dataset.py   Download datasets from Roboflow
│   ├── find_duplicates.py    Group duplicate and near-duplicate images
│   ├── make_splits.py        Build the group-safe split and lock the test set
│   ├── train.py              Fine-tune YOLOv8 models
│   ├── evaluator.py          Evaluate a model or saved predictions
│   ├── open_vocab_predict.py Zero-shot YOLO-World / OWLv2 predictions
│   ├── vlm_predict.py        Zero-shot Qwen2.5-VL predictions
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
| `runs/`    | Ultralytics training artifacts & plots, evaluation reports |
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

Labels are normalized across datasets by `LABEL_MAP` in `src/adma/datasets.py` into these classes (`TARGET_CLASSES`, in class-id order):
`drone`, `military_aircraft`, `military_helicopter`, `military_truck`, `missile`, `missile_launcher`, `tank`, `warship`

Labels mapped to `None` (for example `person`, `Weapons`) are dropped. `military_vehicles` is excluded from the combined split (`COMBINED_DATASET_EXCLUSIONS`) because its generic `Vehicle` class can't be mapped without relabeling.
