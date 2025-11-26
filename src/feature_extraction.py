import os
import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse

RAW_PATH = "data/processed/combined_urls.csv"
OUT_PATH = "data/processed/feature_extracted.csv"
SEED = 42


# ---------------------------------------------------------------------
# Feature extraction for a single URL
# ---------------------------------------------------------------------
def extract_features(url: str) -> dict:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    feats = {}

    # --- Lexical features ---
    feats["url_length"] = len(url)
    feats["hostname_length"] = len(hostname)
    feats["path_length"] = len(path)
    feats["num_dots"] = url.count(".")
    feats["num_hyphens"] = url.count("-")
    feats["num_digits"] = sum(c.isdigit() for c in url)
    feats["num_specials"] = sum(c in "@!$%&*" for c in url)
    feats["num_parameters"] = url.count("=")
    feats["num_slashes"] = url.count("/")

    # --- Domain features ---
    feats["has_ip_address"] = int(bool(re.match(r"(\d{1,3}\.){3}\d{1,3}", hostname)))
    feats["num_subdomains"] = hostname.count(".")
    feats["is_https"] = int(url.startswith("https"))
    feats["tld_length"] = len(hostname.split(".")[-1]) if "." in hostname else 0

    # --- Keyword-based features ---
    keywords = ["login", "verify", "update", "account", "secure",
                "bank", "signin", "paypal", "ebay", "credential"]
    feats["has_suspicious_keyword"] = int(any(k in url.lower() for k in keywords))

    # --- Statistical features ---
    if len(url) > 0:
        alnum_ratio = sum(c.isalnum() for c in url) / len(url)
        entropy = -sum(
            (url.count(c)/len(url)) * np.log2(url.count(c)/len(url))
            for c in set(url)
        )
    else:
        alnum_ratio, entropy = 0, 0
    feats["alnum_ratio"] = round(alnum_ratio, 3)
    feats["entropy"] = round(entropy, 3)

    # --- TLD risk heuristic ---
    risky_tlds = ["tk", "ml", "ga", "cf", "xyz", "top", "gq", "work", "ru", "cn"]
    tld = hostname.split(".")[-1] if "." in hostname else ""
    feats["risky_tld"] = int(tld in risky_tlds)

    # --- Symbolic / positional features ---
    feats["has_at"] = int("@" in url)
    feats["has_double_slash"] = int("//" in url[7:])
    feats["has_equal"] = int("=" in url)
    feats["has_question"] = int("?" in url)
    feats["has_percent"] = int("%" in url)
    feats["has_hash"] = int("#" in url)
    feats["ends_with_slash"] = int(url.endswith("/"))
    feats["starts_with_www"] = int(url.startswith("www."))
    feats["long_url"] = int(len(url) > 75)
    feats["short_url"] = int(len(url) < 25)

    # --- Ratios and combined lengths ---
    feats["path_to_total_ratio"] = round(len(path) / len(url), 3) if len(url) else 0
    feats["hostname_to_total_ratio"] = round(len(hostname) / len(url), 3) if len(url) else 0
    feats["query_length"] = len(query)

    return feats


# ---------------------------------------------------------------------
# Convert all URLs in combined_urls.csv -> feature_extracted.csv
# ---------------------------------------------------------------------
def generate_feature_dataset():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError("Run data_preprocessing.py first to create combined_urls.csv")

    df = pd.read_csv(RAW_PATH)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("combined_urls.csv must contain 'url' and 'label' columns.")

    print(f"[INFO] Extracting features for {len(df):,} URLs...")
    feature_rows = []
    for url in df["url"]:
        try:
            feature_rows.append(extract_features(str(url)))
        except Exception:
            # ensure even bad URLs don't break the run
            feature_rows.append({k: 0 for k in extract_features("http://dummy")})

    features_df = pd.DataFrame(feature_rows)
    final_df = pd.concat([df["url"], features_df, df["label"]], axis=1)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    final_df.to_csv(OUT_PATH, index=False)
    print(f"[SUCCESS] Saved -> {OUT_PATH} ({len(final_df)} rows, {len(final_df.columns)} columns)")
    print("[INFO] Feature columns:", len(features_df.columns))


if __name__ == "__main__":
    generate_feature_dataset()
