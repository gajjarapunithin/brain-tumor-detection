# MRI Brain Tumor Detection

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://github.com/OWNER/brain-tumor-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/OWNER/brain-tumor-detection/actions/workflows/tests.yml)

A PyTorch research project for detecting brain tumors in T1-weighted MRI scans using the BRISC2025 dataset. The pipeline performs tumor-mask segmentation with U-Net, then classifies scans as glioma, meningioma, pituitary, or no tumor using a transfer-learning ResNet50 model.

## Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Results](#results)
- [Technology](#technology)
- [License](#license)

## Features

- Reproducible preprocessing and random seeding
- Binary tumor segmentation with U-Net
- Four-class transfer-learning classification
- Combined segmentation-then-classification inference
- Checkpointing, experiment logs, metrics, plots, and unit tests

## Project Structure

```text
brain-tumor-detection/
├── data/                 # Ignored local dataset and split files
├── notebooks/            # Exploratory notebooks
├── src/                  # Application and research source code
├── configs/              # YAML hyperparameter configurations
├── scripts/              # Thin command-line entry points
├── experiments/          # Local logs and checkpoints
├── results/              # Figures, tables, and prediction overlays
├── tests/                # Unit tests
├── .github/workflows/    # Continuous integration
├── main.py               # Single-image inference CLI
├── requirements.txt      # Pinned runtime dependencies
└── README.md
```

## Installation

Clone the repository, create a virtual environment, and install the pinned dependencies:

```bash
git clone https://github.com/OWNER/brain-tumor-detection.git
cd brain-tumor-detection
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The repository does not include trained weights, medical images, or other large binaries.

## Dataset Setup

Download [BRISC2025 from Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) manually. Kaggle access and download acceptance are required; the dataset is not redistributed in this repository. Extract the dataset under `data/raw/` and inspect the downloaded folder names before training.

The starter loaders expect:

```text
data/raw/classification_task/{train,val,test}/{glioma,meningioma,pituitary,no_tumor}/
data/raw/segmentation_task/{train,val,test}/images/
data/raw/segmentation_task/{train,val,test}/masks/
```

If the archive uses different names, update the discovery logic in `src/data/dataset.py`. Paths are otherwise built relative to the project root by `src/config.py`; no local absolute paths are required.

## Usage

Prepare runtime directories:

```bash
python scripts/run_preprocessing.py
```

Train segmentation and classification independently:

```bash
python scripts/run_train_segmentation.py --config configs/segmentation_config.yaml --run-name segmentation
python scripts/run_train_classification.py --config configs/classification_config.yaml --run-name classification
```

Run both training stages:

```bash
python scripts/run_full_pipeline.py
```

Evaluate trained checkpoints from Python:

```python
from src.evaluation.evaluate_segmentation import evaluate_segmentation
from src.evaluation.evaluate_classification import evaluate_classification

print(evaluate_segmentation("experiments/segmentation/checkpoints/best.pt"))
print(evaluate_classification("experiments/classification/checkpoints/best.pt"))
```

Run inference on one MRI and save a segmentation overlay:

```bash
python main.py path/to/mri.png \
	--segmentation-checkpoint experiments/segmentation/checkpoints/best.pt \
	--classification-checkpoint experiments/classification/checkpoints/best.pt \
	--output results/figures/inference_overlay.png
```

Run tests locally:

```bash
pytest -q tests/
```

## Results

Fill this table after training and evaluation on the selected BRISC2025 split.

| Task | Metric | Validation | Test |
| --- | --- | ---: | ---: |
| Segmentation | Dice | TBD | TBD |
| Segmentation | IoU | TBD | TBD |
| Classification | Accuracy | TBD | TBD |
| Classification | Weighted F1 | TBD | TBD |

## Technology

- Python 3.12+
- PyTorch and torchvision
- OpenCV, NumPy, pandas, scikit-learn
- Matplotlib, seaborn, PyYAML, tqdm
- GitHub Actions for basic test CI

## License

This project is released under the [MIT License](LICENSE). Dataset access and usage remain subject to the BRISC2025/Kaggle dataset terms.
