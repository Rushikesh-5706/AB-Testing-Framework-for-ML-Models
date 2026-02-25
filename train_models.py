"""Train and serialize two ML model variants for A/B testing.

This script trains Model A (Logistic Regression) and Model B (XGBoost) on the
Breast Cancer Wisconsin dataset from scikit-learn. Both models are serialized
to disk along with training metadata including feature names and performance
metrics.
"""

import json
import os

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODELS_DIR = "api/models"
RANDOM_STATE = 42


def train_and_save_models() -> None:
    """Train both model variants and save them to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model A: Logistic Regression ---
    print("Training Model A (Logistic Regression)...")
    model_a = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=1.0,
        solver="lbfgs",
    )
    model_a.fit(X_train_scaled, y_train)

    y_pred_a = model_a.predict(X_test_scaled)
    y_proba_a = model_a.predict_proba(X_test_scaled)[:, 1]

    metrics_a = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_a)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred_a)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba_a)), 4),
    }
    print(f"  Model A Metrics: {metrics_a}")

    from sklearn.pipeline import Pipeline

    pipeline_a = Pipeline([
        ("scaler", scaler),
        ("classifier", model_a),
    ])
    joblib.dump(pipeline_a, os.path.join(MODELS_DIR, "model_A.pkl"))
    print(f"  Saved to {MODELS_DIR}/model_A.pkl")

    # --- Model B: XGBoost ---
    print("Training Model B (XGBoost)...")
    model_b = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model_b.fit(X_train, y_train)

    y_pred_b = model_b.predict(X_test)
    y_proba_b = model_b.predict_proba(X_test)[:, 1]

    metrics_b = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_b)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred_b)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba_b)), 4),
    }
    print(f"  Model B Metrics: {metrics_b}")

    joblib.dump(model_b, os.path.join(MODELS_DIR, "model_B.pkl"))
    print(f"  Saved to {MODELS_DIR}/model_B.pkl")

    # --- Metadata ---
    metadata = {
        "dataset": "Breast Cancer Wisconsin (sklearn built-in)",
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "target_classes": ["malignant (0)", "benign (1)"],
        "random_state": RANDOM_STATE,
        "models": {
            "A": {
                "name": "Logistic Regression",
                "description": "L2-regularized logistic regression with StandardScaler preprocessing",
                "hyperparameters": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
                "metrics": metrics_a,
            },
            "B": {
                "name": "XGBoost",
                "description": "Gradient-boosted decision tree ensemble classifier",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                },
                "metrics": metrics_b,
            },
        },
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")

    # --- Feature sample for testing ---
    sample = {
        "sample_features": X_test[0].tolist(),
        "expected_num_features": len(feature_names),
    }
    sample_path = os.path.join(MODELS_DIR, "sample_input.json")
    with open(sample_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Sample input saved to {sample_path}")

    print("\nModel training complete.")
    print(f"  Model A ({metrics_a['accuracy']:.1%} accuracy) vs "
          f"Model B ({metrics_b['accuracy']:.1%} accuracy)")


if __name__ == "__main__":
    train_and_save_models()
