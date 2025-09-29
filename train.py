# train.py
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.joblib"

def train_and_save():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, n_jobs=None))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"[INFO] Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    payload = {
        "pipeline": pipe,
        "feature_names": list(feature_names)
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"[INFO] Modelo guardado en: {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    train_and_save()
