# Production-Style PyTorch Training Pipeline (MNIST)

This project simulates a simplified ML Platform engineering workflow using a modular PyTorch training pipeline.

## Project structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── tests/
│   └── test_smoke.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## What this pipeline includes

- Dataset preprocessing with normalization for MNIST.
- Deterministic seed setup for reproducibility.
- Modular training and evaluation loops.
- Accuracy metric computation.
- Structured logging to console and file (`logs/ml_pipeline.log`).
- Best-checkpoint persistence to `artifacts/best_model.pt`.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python -m src.main --epochs 3 --batch-size 64 --learning-rate 0.001
```

Optional flags:

- `--data-dir data`
- `--output-dir artifacts`
- `--num-workers 2`
- `--seed 42`
- `--log-level INFO`

## Run with Docker

Build image:

```bash
docker build -t pytorch-mnist-pipeline .
```

Run container:

```bash
docker run --rm pytorch-mnist-pipeline
```

The default container command runs one epoch for a quick smoke run.

## CI workflow

GitHub Actions workflow: `.github/workflows/ci.yml`

On push/PR, CI does:

1. Checkout code
2. Setup Python 3.11
3. Install dependencies from `requirements.txt`
4. Run smoke test (`pytest -q tests/test_smoke.py`)

## Engineering notes

- The codebase is split by concern (`data`, `model`, `train`, `evaluate`, `utils`, `entrypoint`) to mimic production ML systems.
- Logging and reproducibility are first-class to support debugging and repeatable experiments.
- The pipeline can be expanded with experiment tracking, model registry, and deployment orchestration.
