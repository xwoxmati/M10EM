# app.py
import os
import json
import joblib
import logging
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

# ----- Logging básico -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("mlops-api")

# ----- Carga de modelo -----
try:
    payload = joblib.load(MODEL_PATH)
    model = payload["pipeline"]
    feature_names = payload["feature_names"]
    logger.info(f"Modelo cargado. N features: {len(feature_names)}")
except Exception as e:
    logger.exception("No se pudo cargar el modelo.")
    raise e

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True}), 200

def _validate_by_list(features: Any) -> List[float]:
    if not isinstance(features, list):
        raise ValueError("'features' debe ser una lista de números.")
    if len(features) != len(feature_names):
        raise ValueError(f"'features' debe tener {len(feature_names)} valores en orden.")
    try:
        return [float(x) for x in features]
    except Exception:
        raise ValueError("'features' contiene valores no numéricos.")

def _validate_by_named(payload: Dict[str, Any]) -> List[float]:
    if not isinstance(payload, dict):
        raise ValueError("'payload' debe ser un objeto con pares nombre:valor.")
    missing = [f for f in feature_names if f not in payload]
    if missing:
        raise ValueError(f"Faltan features: {missing[:5]}{'...' if len(missing)>5 else ''}")
    try:
        vector = [float(payload[f]) for f in feature_names]
    except Exception:
        raise ValueError("Valores no numéricos en 'payload'.")
    return vector

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True, silent=False)
        if body is None:
            return jsonify({"error": "JSON inválido o ausente."}), 400

        if "features" in body:
            x = _validate_by_list(body["features"])
        elif "payload" in body:
            x = _validate_by_named(body["payload"])
        else:
            return jsonify({
                "error": "Formato inválido. Usa 'features' (lista) o 'payload' (dict con nombres).",
                "expected_feature_count": len(feature_names),
                "feature_names": feature_names
            }), 400

        import numpy as np
        X = np.array(x).reshape(1, -1)
        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)

        return jsonify({
            "prediction": pred,
            "probability": proba,
            "threshold": 0.5,
            "classes": ["malignant(0)", "benign(1)"],
        }), 200

    except ValueError as ve:
        logger.warning(f"Validación fallida: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Error interno en /predict")
        return jsonify({"error": "Error interno del servidor"}), 500

# Manejo global de errores HTTP
@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    return jsonify({"error": e.description}), e.code

if __name__ == "__main__":
    # Para desarrollo local: flask dev server (en Docker usaremos gunicorn)
    app.run(host="0.0.0.0", port=8000, debug=True)
