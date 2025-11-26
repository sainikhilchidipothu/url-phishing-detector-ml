# app/app.py
from flask import Flask, render_template, request
import os
import sys
import joblib
import pandas as pd

# Make `src` importable when running the app directly from the repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from feature_extraction import extract_features

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "app", "templates"))

# Allow overriding model path with an environment variable for flexibility
MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(BASE_DIR, "models", "xgboost_final.pkl")

def load_model(path: str):
    if not os.path.exists(path):
        return None, f"Model not found at {path}. Run `python src/model_training.py` to create models."
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

model, model_err = load_model(MODEL_PATH)



def _get_model_feature_names(m):
    """Return expected feature names for the loaded model (if available)."""
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
    """Map current extractor features to the model's expected feature names.
    Missing features are filled with 0. Simple derived ratios are computed when possible.
    """
    out = {}

    # quick helpers
    def getf(k, default=0):
        return feats.get(k, default)

    url_len = float(getf("url_length", 0)) or 0.0
    num_digits = float(getf("num_digits", 0)) or 0.0
    num_specials = float(getf("num_specials", 0)) or float(getf("num_special", 0) if "num_special" in feats else 0)
    num_hyphens = float(getf("num_hyphens", 0)) or 0.0

    # mapping from model-expected name -> extractor key or derived lambda
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

    for name in expected_names:
        func = mapping.get(name)
        if func is None:
            # fallback to any exact key from feats or 0
            v = feats.get(name, 0)
        else:
            try:
                v = func()
            except Exception:
                v = 0
        # booleans -> int, else numeric coercion
        if isinstance(v, bool):
            out[name] = int(v)
        else:
            try:
                out[name] = float(v)
            except Exception:
                out[name] = 0.0

    return out


def _rule_based_override(feats: dict, compat: dict = None, probs: list | None = None):
    """Small heuristic checks to catch obvious phishing cases that the model may miss.
    Returns None if no override, or a tuple (pred, reason, confidence_override).
    pred: 0 => phishing, 1 => legitimate
    """
    # allow disabling heuristics for testing/production
    try:
        disable = os.environ.get("DISABLE_RULES", "0").lower() in ("1", "true", "yes")
        if disable:
            return None
        # conservative combined heuristics - require one or more strong signals
        num_digits = int(feats.get("num_digits", 0))
        hostname_len = int(feats.get("hostname_length", 0))
        num_subdomains = int(feats.get("num_subdomains", 0))
        has_ip = bool(feats.get("has_ip_address", 0))
        entropy = float(feats.get("entropy", 0) or 0.0)
        risky_tld = int(feats.get("risky_tld", 0) or 0)

        # strong individual signals
        if has_ip:
            return 0, "contains IP address", (100.0 if probs is None else round(max(probs[0]*100, 99.0), 2))

        if risky_tld:
            # some TLDs are more frequently used in phishing; treat as a strong signal
            return 0, "risky TLD", (95.0 if probs is None else round(max(probs[0]*100, 90.0), 2))

        # combine signals: digits + short path/short url or high entropy
        signals = 0
        if num_digits >= 3:
            signals += 1
        if num_subdomains >= 3:
            signals += 1
        if hostname_len > 20 and num_digits >= 2:
            signals += 1
        if entropy > 4.5:
            signals += 1
        # short URL with digits is suspicious too
        short_url = bool(feats.get("short_url", 0))
        if short_url and num_digits >= 2:
            signals += 1

        # if two or more signals, consider phishing
        if signals >= 2:
            return 0, "combined suspicious signals", (95.0 if probs is None else round(max(probs[0]*100, 90.0), 2))

    except Exception:
        pass
    return None


# Cache model expected feature names at startup for consistent use and logging
MODEL_EXPECTED_NAMES = _get_model_feature_names(model) if model is not None else None
print(f"[INFO] Loaded model. expected feature names: {MODEL_EXPECTED_NAMES}")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    url = ""

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if not url:
            prediction = "⚠️ Please provide a URL."
        elif model is None:
            prediction = f"⚠️ Model error: {model_err}"
        else:
            # --- Normalize input so users can paste plain text or incomplete URLs ---
            normalized = url
            # replace whitespace with '-' for safety, but keep characters otherwise
            if " " in normalized:
                normalized = normalized.replace(" ", "-")
            if not normalized.startswith(("http://", "https://")):
                normalized = "http://" + normalized

            try:
                feats = extract_features(normalized)

                # If the model exposes expected feature names, build compat features and
                # predict immediately using that 1-row DataFrame. This avoids any
                # later reindexing to a different feature set which causes shape mismatch.
                expected = MODEL_EXPECTED_NAMES
                if expected:
                    expected_str = [str(n) for n in expected]
                    compat = _build_compat_features(feats, expected_str)
                    X = pd.DataFrame([compat], columns=expected_str)
                    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
                    # immediate prediction path
                    try:
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X)[0]
                            pred = int((probs[1] > 0.5))
                            # rule-based override: allow heuristics to flip prediction for obvious phishing
                            override = _rule_based_override(feats, compat, probs.tolist())
                            if override is not None:
                                pred, reason, conf_override = override
                                prediction = "🚨 Phishing URL Detected!" if pred == 0 else "✅ Legitimate URL"
                                confidence = conf_override
                            else:
                                prediction = "✅ Legitimate URL" if pred == 1 else "🚨 Phishing URL Detected!"
                                confidence = round(float(probs[1] * 100) if pred == 1 else float(probs[0] * 100), 2)
                        else:
                            pred = int(model.predict(X)[0])
                            prediction = "✅ Legitimate URL" if pred == 1 else "🚨 Phishing URL Detected!"
                    except Exception as e:
                        print(f"[ERROR] Immediate predict failed: {e}")
                        prediction = f"⚠️ Error processing input. Make sure you provided a valid URL or short text. ({e})"
                    # render result (skip remaining alignment/scaler logic)
                    return render_template("index.html", prediction=prediction, confidence=confidence, url=url)
                else:
                    X = pd.DataFrame([feats])

                # If we reached here, we did not take the immediate-predict path; try aligning to processed CSV order
                if True:
                    feat_csv = os.path.join(BASE_DIR, "data", "processed", "feature_extracted.csv")
                    if os.path.exists(feat_csv):
                        try:
                            sample_df = pd.read_csv(feat_csv, nrows=1)
                            train_feat_cols = [c for c in sample_df.columns if c not in ("url", "label")]
                            X = X.reindex(columns=train_feat_cols, fill_value=0)
                            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
                        except Exception as e:
                            print(f"[WARN] Could not align features with training CSV: {e}")

                # If a scaler was saved during training, apply it when available
                scaler_path = os.environ.get("SCALER_PATH") or os.path.join(BASE_DIR, "models", "scaler.pkl")
                if os.path.exists(scaler_path):
                    try:
                        scaler = joblib.load(scaler_path)
                        if hasattr(scaler, "transform"):
                            # scaler expects same column order as training; assume X is ordered
                            X_scaled = scaler.transform(X)
                            X_for_pred = X_scaled
                        else:
                            X_for_pred = X
                    except Exception:
                        X_for_pred = X
                else:
                    X_for_pred = X

                # Predict
                # Final normalization: ensure X_for_pred matches model expected names exactly
                if MODEL_EXPECTED_NAMES is not None:
                    expected_str = [str(n) for n in MODEL_EXPECTED_NAMES]
                    try:
                        # If it's a numpy array (from scaler), convert back to DataFrame using current X columns
                        if not isinstance(X_for_pred, pd.DataFrame):
                            # attempt to reconstruct from X (the DataFrame before scaling)
                            try:
                                X_for_pred = pd.DataFrame(X_for_pred, columns=X.columns)
                            except Exception:
                                # fallback: create empty DF with expected cols
                                X_for_pred = pd.DataFrame(columns=expected_str)
                        # Reindex to expected columns (drop extras, fill missing)
                        X_for_pred = X_for_pred.reindex(columns=expected_str, fill_value=0)
                        X_for_pred = X_for_pred.apply(pd.to_numeric, errors="coerce").fillna(0)
                    except Exception as e:
                        print(f"[ERROR] Final normalization failed: {e}")

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_for_pred)[0]
                    # default class mapping in training: 0=phishing,1=legitimate
                    pred = int((probs[1] > 0.5))
                    override = _rule_based_override(feats, None, probs.tolist())
                    if override is not None:
                        pred, reason, conf_override = override
                        prediction = "🚨 Phishing URL Detected!" if pred == 0 else "✅ Legitimate URL"
                        confidence = conf_override
                    else:
                        prediction = "✅ Legitimate URL" if pred == 1 else "🚨 Phishing URL Detected!"
                        confidence = round(float(probs[1] * 100) if pred == 1 else float(probs[0] * 100), 2)
                else:
                    pred = int(model.predict(X_for_pred)[0])
                    prediction = "✅ Legitimate URL" if pred == 1 else "🚨 Phishing URL Detected!"

            except Exception as e:
                # Log diagnostic info for easier debugging
                try:
                    cols = list(X.columns) if isinstance(X, pd.DataFrame) else None
                except Exception:
                    cols = None
                print(f"[ERROR] Prediction failed: {e}")
                print(f"[ERROR] Input columns ({None if cols is None else len(cols)}): {cols}")
                print(f"[ERROR] Model expected names: {MODEL_EXPECTED_NAMES}")
                # Return a concise message to the user
                prediction = f"⚠️ Error processing input. Make sure you provided a valid URL or short text. ({e})"

    return render_template("index.html", prediction=prediction, confidence=confidence, url=url)


if __name__ == "__main__":
    # Disable the interactive debugger to avoid Werkzeug's authorization page (403).
    # For local development you can set debug=False; enable debug only when needed.
    import socket

    desired_port = int(os.environ.get("PORT", "5000"))
    bind_port = desired_port

    def _port_free(p):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return True
        except OSError:
            return False

    if not _port_free(bind_port):
        # pick a free ephemeral port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        bind_port = s.getsockname()[1]
        s.close()
        print(f"[WARN] Port {desired_port} was in use, starting on available port {bind_port} instead")

    print(f"[INFO] Starting Flask app on 127.0.0.1:{bind_port}")
    try:
        app.run(host="127.0.0.1", port=bind_port, debug=False)
    except Exception as e:
        print(f"[ERROR] Failed to start Flask app: {e}")
