# MRI Brain Tumor Detection

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
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
├── requirements-core.txt # Non-framework runtime dependencies
├── requirements-cpu.txt  # Optional official CPU-only PyTorch pins
└── README.md
```

## Installation

Use Python 3.11 (verified for this project), create a virtual environment, then install the supported CPU build and core packages. PowerShell:

```bash
& 'C:\Program Files\Python311\python.exe' -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements-cpu.txt
venv\Scripts\python.exe -m pip install -r requirements-core.txt -r requirements-dev.txt
```

The environment's framework build must match the target accelerator. For CUDA, MPS, or another supported backend, install a compatible `torch`/`torchvision` pair using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for that OS and hardware, then install `requirements-core.txt`. Do not install CUDA packages on a CPU-only host or assume an accelerator works until the runtime check succeeds. The application never downloads or switches packages at startup.

## Hardware and resource profiles

The shared hardware service probes a tiny tensor operation before reporting CUDA, MPS, or XPU as operational. `auto` tries available PyTorch accelerators and falls back to CPU with a reason. `cpu` forces CPU for debugging/reproducibility; an explicitly requested unavailable accelerator returns an error. The project does not promise universal AMD or Intel acceleration.

```powershell
venv\Scripts\python.exe -m scripts.check_hardware
venv\Scripts\python.exe -m scripts.check_hardware --device cpu --resource-profile low_memory --batch-size 8 --num-workers 2
venv\Scripts\python.exe -m scripts.benchmark_training --device auto --resource-profile auto
```

Training YAML supports `resource_profile: auto|low_memory|balanced|high_performance|custom`, `device`, `batch_size`, `num_workers`, `gradient_accumulation_steps`, `mixed_precision`, and `pin_memory`. Low-memory mode reduces per-step batch allocation and uses accumulation; image resolution and model architecture stay as configured. `mixed_precision: auto` conservatively selects float32; explicitly select `float16` or `bfloat16` only after a representative benchmark and smoke test. Pinned memory and non-blocking copies are used only for operational CUDA. Profiles do not assume a larger GPU needs a larger batch.

Single-image inference loads checkpoints once, uses inference mode, and uses the same configured resize/normalization as training. On CUDA out-of-memory it retries on CPU, moves both models, and returns/reports an explicit fallback message. A predicted mask is a localization estimate and does not by itself establish that a tumor is present; consult the separate class prediction as well. Training never switches device mid-run; the latest completed epoch checkpoint remains available for resuming after an OOM.

## Dataset Setup

The already provisioned BRISC2025 dataset is expected at `DATASETS/mri/brisc2025`. The source dataset is read-only; the loaders use its manifests and official splits.

The starter loaders expect:

```text
DATASETS/mri/brisc2025/classification_task/{train,test}/{glioma,meningioma,pituitary,no_tumor}/
DATASETS/mri/brisc2025/segmentation_task/{train,test}/images/ and masks/
```

Validation subsets are derived reproducibly from training data; the official test split is reserved for evaluation.

## Usage

### Desktop application (optional)

The desktop workstation is an optional presentation layer around the existing inference command. Install its small Qt Widgets dependency into the validated environment; this does not replace the ML packages:

```powershell
venv_cuda\Scripts\python.exe -m pip install -r requirements-ui.txt
venv_cuda\Scripts\python.exe -m app.main
```

Choose or drop a JPG, PNG, BMP, or TIFF MRI, then run analysis. The application runs the same full-image classifier and segmentation pipeline as the CLI in a background process. Review the original image, generated mask, and overlay in separate tabs or side-by-side; use the zoom controls to inspect them. The result panel shows the actual four class softmax scores (not clinically calibrated probabilities) and mask foreground fraction (not tumor probability). No Dice/IoU is shown for an individual scan because there is no ground-truth mask at inference time.

The NeuroScan AI app icon is bundled at `app/neuroscan_icon.png` and appears in the application window and sidebar branding.

Use **Save Overlay**, **Save Mask**, or **Export JSON** to copy the generated artifacts to a chosen location. **Open Results Folder** opens the run directory. Completed analyses are also listed in the local history page. Generated output folders are under `results/ui_analyses/`; the small local history and application log are stored there too. Input scans and model checkpoints are never moved, and the application does not transmit data externally.

The application is a research-use prototype. AI predictions can be incorrect and are not a medical diagnosis; results should be interpreted by qualified medical professionals. Segmentation provides a localization estimate and does not by itself establish that a tumor is present.

For CLI-only use or environments without a graphical desktop, continue using the `scripts.run_full_pipeline` command below.

Train segmentation and classification independently:

```powershell
venv_cuda\Scripts\python.exe -m scripts.run_train_segmentation --config configs\segmentation_config.yaml --run-name segmentation --device cuda --resource-profile low_memory --mixed-precision float16
venv_cuda\Scripts\python.exe -m scripts.run_train_classification --config configs\classification_config.yaml --run-name classification --device cuda --resource-profile low_memory --mixed-precision float16
```

Classification training uses `configs/classification_duplicate_exclusions.json`. It filters training copies of exact-content images appearing in the official test split before making the seeded train/validation split. Original dataset images and the official test split are unchanged. For the final classification metrics, evaluate the classification model by itself (the official test is evaluated once):

```powershell
venv_cuda\Scripts\python.exe -m scripts.evaluate_classification_test --checkpoint experiments\classification\checkpoints\best.pt --output-dir results\classification_test --device cuda
```

Run integrated inference on one MRI. The classifier receives the full image, matching its training preprocessing. The command saves an overlay, aligned binary mask, and JSON result containing class scores (not clinically calibrated probabilities):

```powershell
venv_cuda\Scripts\python.exe -m scripts.run_full_pipeline path\to\mri.png `
  --segmentation-checkpoint experiments\segmentation\checkpoints\best.pt `
  --classification-checkpoint experiments\classification\checkpoints\best.pt `
  --output results\inference\overlay.png --device cuda --full-image
```

Run tests locally:

```bash
venv_cuda\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp
```

## Results

| Task | Metric | Validation | Test |
| --- | --- | ---: | ---: |
| Segmentation | Dice | See experiment log | 0.827753 |
| Segmentation | IoU | See experiment log | 0.748113 |
| Classification | Accuracy | 0.963904 (selected epoch 20) | 0.939000 |
| Classification | Macro F1 | — | 0.931074 |
| Classification | Weighted F1 | — | 0.939961 |

Classification test details, per-class metrics, and confusion matrices are in `results/classification_test/`. The official split contains five within-test exact-image duplicate groups; all 1,000 official samples were included exactly once. Nine training-side copies overlapping test image content were excluded using the audit manifest. Test metrics are not used for model selection.

## Technology

- Python 3.11 (verified; compatible Python versions depend on matching wheel availability)
- PyTorch and torchvision
- OpenCV, NumPy, pandas, scikit-learn
- Matplotlib, seaborn, PyYAML, tqdm
- GitHub Actions for basic test CI

- ## Security Review and Vulnerability Assessment

AI-assisted tools may be used during development to help identify potential security vulnerabilities, detect insecure coding practices, and highlight areas of the codebase that may require further review. These tools serve as an additional layer of support in the security review process and do not replace manual code review or testing.

Any identified issues should be carefully reviewed and verified before corrective measures are implemented. While AI-assisted analysis can help improve security awareness and identify potential risks, it cannot guarantee that all vulnerabilities will be detected or that the application is completely secure.

Regular testing, dependency checks, and code reviews remain important parts of maintaining the security and reliability of the project.

## License

This project is released under the [MIT License](LICENSE). Dataset access and usage remain subject to the BRISC2025 dataset terms.
