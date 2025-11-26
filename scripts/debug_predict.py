"""Simple debug harness: reproduce model input and prediction for a given URL.
Run with the project venv Python.
"""
import json
import importlib
import os
import sys

# Make project root importable so `importlib.import_module('app.app')` works
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODULE = 'app.app'
URL = 'www.af3ooaalalf9r99f.com'

def main():
    m = importlib.import_module(MODULE)
    url = URL
    # normalize same as the app
    if ' ' in url:
        url = url.replace(' ', '-')
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    print('Normalized URL:', url)
    feats = m.extract_features(url)
    print('\nEXTRACTED FEATURES:')
    print(json.dumps(feats, indent=2))

    expected = getattr(m, 'MODEL_EXPECTED_NAMES', None)
    print('\nMODEL_EXPECTED_NAMES:', expected)

    if expected:
        expected_str = [str(n) for n in expected]
        compat = m._build_compat_features(feats, expected_str)
        import pandas as pd
        X = pd.DataFrame([compat], columns=expected_str)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        print('\nCOMPAT FEATURE VECTOR:')
        print(X.to_dict(orient='records')[0])

        model = m.model
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[0]
                print('\nMODEL PROBS:', probs.tolist())
                pred = int((probs[1] > 0.5))
                print('MODEL PRED:', pred)
                # show rule-based override if available
                try:
                    override = getattr(m, '_rule_based_override')(feats, compat, probs.tolist())
                    print('RULE-BASED OVERRIDE:', override)
                except Exception as _:
                    pass
            else:
                print('MODEL has no predict_proba; pred:', int(model.predict(X)[0]))
        except Exception as e:
            print('PREDICTION ERROR:', e)
    else:
        print('Model does not expose expected feature names. Cannot run compat path.')

if __name__ == '__main__':
    main()
