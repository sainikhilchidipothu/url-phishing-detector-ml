"""
scripts/smoke_test.py
Quick check: load the XGBoost model and run feature extraction on one sample URL.
"""
import os
import sys
import joblib
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(BASE_DIR, 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from feature_extraction import extract_features

MODEL_PATH = os.environ.get('MODEL_PATH') or os.path.join(BASE_DIR, 'models', 'xgboost_final.pkl')

print('Model path:', MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    print('Model file not found. Run: python src/model_training.py')
    raise SystemExit(1)

try:
    model = joblib.load(MODEL_PATH)
    print('Model loaded. predict_proba available:', hasattr(model, 'predict_proba'))
except Exception as e:
    print('Failed to load model:', e)
    raise

sample_url = 'https://example.com/login'
feats = extract_features(sample_url)
 

def _get_model_feature_names(m):
    try:
        if hasattr(m, "feature_names_in_"):
            return list(m.feature_names_in_)
        if hasattr(m, "get_booster"):
            booster = m.get_booster()
            if getattr(booster, "feature_names", None):
                return list(booster.feature_names)
    except Exception:
        pass
    return None


def _build_compat_features(feats: dict, expected_names: list):
    out = {}
    def getf(k, default=0):
        return feats.get(k, default)

    url_len = float(getf("url_length", 0)) or 0.0
    num_digits = float(getf("num_digits", 0)) or 0.0
    num_specials = float(getf("num_specials", 0)) or float(getf("num_special", 0) if "num_special" in feats else 0)
    num_hyphens = float(getf("num_hyphens", 0)) or 0.0

    mapping = {
        "url_length": lambda: url_len,
        "domain_length": lambda: float(getf("hostname_length", 0)),
        "path_length": lambda: float(getf("path_length", 0)),
        "num_digits": lambda: num_digits,
        "num_special": lambda: num_specials,
        "entropy": lambda: float(getf("entropy", 0)),
        "num_subdomains": lambda: float(getf("num_subdomains", 0)),
        "has_ip": lambda: int(bool(getf("has_ip_address", 0))),
        "has_https": lambda: int(bool(getf("is_https", 0))),
        "count_double_slash": lambda: int(bool(getf("has_double_slash", 0))),
        "has_at": lambda: int(bool(getf("has_at", 0))),
        "has_dash_domain": lambda: int(num_hyphens > 0),
        "is_shortened": lambda: int(bool(getf("short_url", 0))),
        "has_susp_word": lambda: int(bool(getf("has_suspicious_keyword", 0))),
        "has_brand": lambda: 0,
        "ratio_digits": lambda: (num_digits / url_len) if url_len else 0.0,
        "ratio_special": lambda: (num_specials / url_len) if url_len else 0.0,
    }
    # build output dict in the order of expected_names
    for name in expected_names:
        func = mapping.get(name)
        if func is None:
            v = feats.get(name, 0)
        else:
            try:
                v = func()
            except Exception:
                v = 0
        if isinstance(v, bool):
            out[name] = int(v)
        else:
            try:
                out[name] = float(v)
            except Exception:
                out[name] = 0.0

    return out

expected = _get_model_feature_names(model)
if expected:
    compat = _build_compat_features(feats, expected)
    X = pd.DataFrame([compat], columns=expected)
else:
    X = pd.DataFrame([feats])

print('Feature columns count (after compatibility):', X.shape[1])
try:
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[0]
        pred = int((probs[1] > 0.5))
        print('Sample prediction for', sample_url, '->', pred, 'confidence:', round(probs[1]*100,2))
    else:
        pred = model.predict(X)[0]
        print('Sample prediction for', sample_url, '->', pred)
except Exception as e:
    print('Prediction failed:', e)
    raise
