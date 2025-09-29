# tests/test_train.py
from pathlib import Path
import joblib
import train

def test_train_and_save(tmp_path, monkeypatch):
    # redirige models/ a tmp_path para no ensuciar repo
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "model.joblib")
    train.train_and_save()
    p = tmp_path / "model.joblib"
    assert p.exists()
    payload = joblib.load(p)
    assert "pipeline" in payload and "feature_names" in payload
