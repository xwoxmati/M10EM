# tests/test_api_smoke.py
import json
import app

def test_health():
    client = app.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("status") == "ok"

def test_predict_bad_body():
    client = app.app.test_client()
    resp = client.post("/predict", json={"foo": "bar"})
    assert resp.status_code == 400

def test_predict_minimal_list():
    client = app.app.test_client()
    # Usa longitud correcta de features:
    n = len(app.feature_names)
    resp = client.post("/predict", json={"features": [0.0]*n})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "prediction" in data and "probability" in data
