# Phishing URL Detector — Simple Run Guide

This repository contains a small machine-learning pipeline and a Flask app that detects phishing URLs using feature-based models (XGBoost by default).

Quick steps to run locally
1. Create and activate a Python virtual environment (macOS / zsh):

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare data (creates `data/processed/combined_urls.csv`):

```bash
python src/data_preprocessing.py
```

4. Extract features (creates `data/processed/feature_extracted.csv`):

```bash
python src/feature_extraction.py
```

5. Train models (saves models into `models/` and results into `results/`):

```bash
python src/model_training.py
```

6. Run the Flask app (uses `models/xgboost_final.pkl` by default):

```bash
python app/app.py
```

If you want to point the app to a different model file, set `MODEL_PATH` environment variable before running.

Quick smoke test

```bash
python scripts/smoke_test.py
```

If anything fails, check the console messages. Typical fixes: missing model files (run training), missing processed data (run preprocessing and feature extraction), or unmet dependencies (install from `requirements.txt`).

That's it — the rest of the code is intentionally small and documented in the `src/` folder.
