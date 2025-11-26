import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------
def metrics(y_true, y_pred):
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


# ---------------------------------------------------------------------
# Main training and evaluation pipeline
# ---------------------------------------------------------------------
def train_and_eval():
    in_path = "data/processed/feature_extracted.csv"
    models_dir, results_dir = "models", "results"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- Load Data ---
    if not os.path.exists(in_path):
        raise FileNotFoundError("Run feature_extraction.py first.")
    df = pd.read_csv(in_path)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df.drop(columns=[c for c in df.columns if df[c].dtype == "object" and c != "label"])
    df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)

    X, y = df.drop(columns=["label"]), df["label"]
    if len(y.unique()) < 2:
        raise ValueError("Dataset still has one class. Check preprocessing step.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("[INFO] Original train distribution:\n", y_train.value_counts(normalize=True))

    # --- Balance data using SMOTE ---
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print("[INFO] After SMOTE balancing:", pd.Series(y_train).value_counts().to_dict())

    # --- Class Weights ---
    class_weights_rf = {0: 1, 1: 3}
    class_weights_svm = {0: 1, 1: 4}

    # --- Define Models ---
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=3,
        class_weight=class_weights_rf,
        random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma=0.9,
        class_weight=class_weights_svm,
        probability=True,
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.5,
        scale_pos_weight=0.6,
        use_label_encoder=False,
        random_state=42
    )

    # --- Train Models ---
    print("[INFO] Training models...")
    rf.fit(X_train, y_train)
    svm.fit(scaler.transform(X_train), y_train)
    xgb.fit(X_train, y_train, verbose=False)

    # --- Evaluate ---
    thresholds = {"Random Forest": 0.5, "SVM": 0.5, "XGBoost": 0.5}
    results = []

    for name, model, X_eval in [
        ("Random Forest", rf, X_test),
        ("SVM", svm, scaler.transform(X_test)),
        ("XGBoost", xgb, X_test)
    ]:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_eval)[:, 1]
            preds = (probs > thresholds[name]).astype(int)
        else:
            preds = model.predict(X_eval)
        results.append({"Model": name, **metrics(y_test, preds)})

    res = pd.DataFrame(results)
    print("\n=== FINAL TEST RESULTS ===\n", res.to_string(index=False))

    # --- Save multiple CSVs ---
    res.to_csv(os.path.join(results_dir, "results.csv"), index=False)

    rounded = res.copy()
    rounded[["accuracy", "precision", "recall", "f1"]] = rounded[["accuracy", "precision", "recall", "f1"]].round(3)
    rounded.to_csv(os.path.join(results_dir, "model_performance.csv"), index=False)

    summary = rounded.copy()
    summary.loc["Mean"] = summary.mean(numeric_only=True)
    summary.to_csv(os.path.join(results_dir, "final_model_performance.csv"), index=False)
    print("[SAVED] model_performance.csv and final_model_performance.csv generated successfully.")

    # -----------------------------------------------------------------
    # Clean Visualization: Model Comparison
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    res_melted = res.melt(id_vars="Model", var_name="Metric", value_name="Score")

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    ax = sns.barplot(
        data=res_melted,
        x="Model", y="Score", hue="Metric",
        palette=colors, edgecolor="black"
    )

    plt.title("Model Performance Comparison", fontsize=16, weight="bold", pad=20)
    plt.ylabel("Score", fontsize=13)
    plt.xlabel("Model", fontsize=13)
    plt.ylim(0.6, 1.0)

    plt.legend(
        title="Metric",
        fontsize=10,
        title_fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        frameon=True
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=2, color="black")

    plt.tight_layout(pad=2.5)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------
    # Confusion Matrices for All Models
    # -----------------------------------------------------------------
    print("\n[INFO] Generating confusion matrices for all models...")

    model_dict = {
        "Random Forest": (rf, X_test),
        "SVM": (svm, scaler.transform(X_test)),
        "XGBoost": (xgb, X_test)
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (name, (model, X_eval)) in enumerate(model_dict.items()):
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_eval)[:, 1]
            preds = (probs > thresholds[name]).astype(int)
        else:
            preds = model.predict(X_eval)

        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
            xticklabels=["Phishing", "Legitimate"],
            yticklabels=["Phishing", "Legitimate"],
            cbar=(i == 2)
        )
        axes[i].set_title(f"{name}", fontsize=11, weight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    fig.suptitle("Confusion Matrices for All Models", fontsize=14, weight="bold")
    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    print("\nClassification Reports:")
    for name, (model, X_eval) in model_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_eval)[:, 1]
            preds = (probs > thresholds[name]).astype(int)
        else:
            preds = model.predict(X_eval)
        print(f"\n--- {name} ---\n", classification_report(y_test, preds, digits=4))

    # --- Save Models ---
    joblib.dump(rf, os.path.join(models_dir, "rf.pkl"))
    joblib.dump(svm, os.path.join(models_dir, "svm.pkl"))
    joblib.dump(xgb, os.path.join(models_dir, "xgb.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print(f"[SAVED] Models -> {models_dir}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    train_and_eval()
