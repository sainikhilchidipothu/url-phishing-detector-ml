import os
import pandas as pd
from pathlib import Path

# ===== Base Settings =====
PHISH_FRACTION_DEFAULT = 0.7   # target ratio if legit data is big enough
PER_CLASS_CAP = 20500           # max rows per class (keeps size manageable)
MIN_PER_CLASS = 7000            # minimum per class after balancing
SEED = 42
# ==========================

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHISH_PATH = RAW_DIR / "phishtank_raw.csv"
LEGIT_PATH = RAW_DIR / "legitimate_urls.csv"
OUT_COMBINED = OUT_DIR / "combined_urls.csv"


def _read_urls_csv(path: Path) -> pd.DataFrame:
    """Read a CSV and return one 'url' column."""
    if not path.exists():
        print(f"[WARN] {path} not found.")
        return pd.DataFrame(columns=["url"])
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)
    cols = [c for c in df.columns if any(k in str(c).lower() for k in ["url", "link", "domain"])]
    col = cols[0] if cols else df.columns[0]
    df = df[[col]].rename(columns={col: "url"})
    df["url"] = df["url"].astype(str).str.strip()
    df["url"] = df["url"].apply(lambda u: u if u.startswith(("http://", "https://")) else f"http://{u}")
    df = df[df["url"].str.contains(r"[A-Za-z0-9\-_.]+\.[A-Za-z]{2,}", na=False)]
    return df.drop_duplicates(subset="url").reset_index(drop=True)


def _default_legit() -> pd.DataFrame:
    """Fallback legitimate list if no file exists."""
    urls = [
        "https://google.com","https://wikipedia.org","https://github.com",
        "https://bbc.com","https://nytimes.com","https://microsoft.com",
        "https://apple.com","https://amazon.com","https://linkedin.com","https://reddit.com"
    ]
    return pd.DataFrame({"url": urls})


def preprocess_data():
    print("[INFO] Loading phishing data...")
    phish = _read_urls_csv(PHISH_PATH)
    phish["label"] = 0
    print(f"[INFO] Phishing URLs loaded: {len(phish):,}")

    print("[INFO] Loading legitimate data...")
    legit = _read_urls_csv(LEGIT_PATH)
    if legit.empty:
        print("[WARN] Legitimate dataset empty — using default fallback URLs.")
        legit = _default_legit()
    legit["label"] = 1
    print(f"[INFO] Legitimate URLs loaded: {len(legit):,}")

    if phish.empty:
        raise ValueError("Phishing dataset is empty — check phishtank_raw.csv")

    # === Adaptive phishing ratio ===
    legit_count = len(legit)
    phish_count = len(phish)
    phish_fraction = PHISH_FRACTION_DEFAULT

    # Adjust dynamically if legit data is too small
    if legit_count < 100:
        phish_fraction = 0.95
    elif legit_count < 500:
        phish_fraction = 0.9
    elif legit_count < 2000:
        phish_fraction = 0.88
    elif legit_count < 5000:
        phish_fraction = 0.85
    else:
        phish_fraction = PHISH_FRACTION_DEFAULT

    print(f"[INFO] Adaptive phishing fraction set to {phish_fraction:.2f}")

    # --- Compute target sizes ---
    total_target = min(phish_count + legit_count, PER_CLASS_CAP * 2)
    target_phish = int(total_target * phish_fraction)
    target_legit = total_target - target_phish

    target_phish = max(min(target_phish, PER_CLASS_CAP), MIN_PER_CLASS)
    target_legit = max(min(target_legit, PER_CLASS_CAP), MIN_PER_CLASS)

    # --- Sample or upsample to targets ---
    def _sample(df, n, label):
        if len(df) >= n:
            return df.sample(n=n, random_state=SEED)
        print(f"[INFO] Upsampling {label} from {len(df)} -> {n}")
        return df.sample(n=n, replace=True, random_state=SEED)

    phish_bal = _sample(phish, target_phish, "phishing")
    legit_bal = _sample(legit, target_legit, "legitimate")

    # Combine & shuffle
    combined = pd.concat([phish_bal, legit_bal], axis=0)
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    combined.to_csv(OUT_COMBINED, index=False)
    print(f"[SUCCESS] Saved combined dataset -> {OUT_COMBINED}")
    print("[INFO] Final label distribution:\n", combined["label"].value_counts())

if __name__ == "__main__":
    preprocess_data()
