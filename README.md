# MLOps en la Nube — Breast Cancer API (Evaluación Modular 10)

API REST para predecir diagnóstico (benigno/maligno) usando el dataset **Breast Cancer Wisconsin (Diagnostic)**. El repositorio incluye entrenamiento, serialización del modelo, servicio Flask, contenedor Docker y CI con GitHub Actions (build, tests y smoke test).

---

## 1) Resumen

- **Modelo**: Regresión Logística con `StandardScaler` en `Pipeline`.
- **Datos**: Breast Cancer Wisconsin (scikit-learn / Kaggle).
- **Serialización**: `joblib` → `models/model.joblib`.
- **API**: Flask con `GET /` (health) y `POST /predict` (validación por lista o por nombres de features).
- **Contenedores**: `Dockerfile` (imagen slim) con `gunicorn`.
- **CI/CD**: GitHub Actions que instala dependencias, entrena (si falta el modelo), ejecuta `pytest`, construye la imagen y hace smoke tests.

---

## 2) Arquitectura (alto nivel)

```
Cliente → (JSON) → Flask API → Pipeline (Scaler + LogisticRegression) → Predicción JSON
                         │
                         └── joblib: models/model.joblib
```

---

## 3) Estructura del repositorio

```
.
├─ app.py
├─ train.py
├─ requirements.txt
├─ Dockerfile
├─ README.md
├─ .gitignore
├─ models/
│  └─ model.joblib         # se genera al entrenar (opcional versionar)
├─ tests/
│  ├─ test_train.py
│  └─ test_api_smoke.py
└─ .github/
   └─ workflows/
      └─ ci.yml
```

---

## 4) Requisitos

- Python 3.11+ (recomendado entorno virtual)
- Windows/PowerShell o Linux/Mac
- (Opcional) Docker Desktop
- Git + cuenta GitHub

---

## 5) Instalación y entrenamiento (local)

**Windows PowerShell** (en la carpeta del proyecto):

```powershell
# Crear y usar venv (si no activas, puedes usar .venv\Scripts\python.exe en cada comando)
python -m venv .venv
# Si PowerShell bloquea scripts, ejecuta en una PowerShell con permisos de usuario:
# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Entrenar y guardar el modelo en models/model.joblib
python train.py
```

Salida esperada (aprox):
```
[INFO] Accuracy: 0.9xxx | ROC-AUC: 0.9xxx
[INFO] Modelo guardado en: ...\models\model.joblib
```

> **Nota**: En CI, si no existe `models/model.joblib`, se entrena automáticamente.

---

## 6) Ejecutar la API (local)

```powershell
# Con el venv activado:
python app.py
# Servirá en: http://127.0.0.1:8000/
```

Probar endpoints (**PowerShell**):

```powershell
# Health
curl.exe "http://127.0.0.1:8000/"

# Predicción con lista de 30 valores (ejemplo con ceros)
curl.exe -s -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  --data-raw '{"features":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```

Alternativa PowerShell nativa:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"features":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```

También acepta **payload con nombres de features** (el orden da igual):

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body

---

## 7) Docker

Construir y correr:

```powershell
docker build -t mlops-bc:latest .
docker run --rm -p 8000:8000 mlops-bc:latest
```

Probar:

```powershell
curl.exe "http://127.0.0.1:8000/"
```

---

## 8) Tests (pytest)

```powershell
pytest -q
```

Incluye:
- `tests/test_train.py`: verifica entrenamiento/serialización.
- `tests/test_api_smoke.py`: health y predicción mínima.

---

## 9) CI/CD con GitHub Actions

Workflow en `.github/workflows/ci.yml` que hace:

1. Setup Python + dependencias  
2. Entrena si falta `models/model.joblib`  
3. Ejecuta `pytest`  
4. `docker build`  
5. Levanta contenedor y valida `GET /` + `POST /predict` (smoke test)

Se ejecuta automáticamente en **push/pull request** y también puede correrse manualmente si habilitas `workflow_dispatch`.

---

## 10) Variables de entorno

- `MODEL_PATH` (opcional): ruta al modelo. Por defecto `models/model.joblib`.

---

## 11) Mapeo a la Rúbrica (guía rápida)

- **Modelado y Serialización**: `train.py` entrena, evalúa (Accuracy/ROC-AUC) y guarda con `joblib`.
- **API REST**: `app.py` con endpoints, validación de entrada (lista o dict con nombres), logging y manejo de errores.
- **Contenedores**: `Dockerfile` slim con `gunicorn`, puerto 8000.
- **Pruebas**: `pytest` con tests de entrenamiento y smoke tests API.
- **CI/CD**: `ci.yml` con build, tests y smoke tests automáticos.
- **Documentación**: este `README` explica instalación, uso, Docker, tests, CI, variables y mapeo de la rúbrica.
---

## 12) Referencias

- **Dataset**
  - Kaggle — Breast Cancer Wisconsin (Diagnostic): https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
  - Scikit-learn — `load_breast_cancer`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

- **Librerías y herramientas**
  - Scikit-learn, Flask, Gunicorn, joblib, NumPy, Pytest
  - Docker, GitHub Actions

---

## 13) Comandos rápidos (Windows PowerShell)

```powershell
# 1) Entorno + deps + train
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python train.py

# 2) Ejecutar API
python app.py
# (en otra terminal) curl.exe "http://127.0.0.1:8000/"

# 3) Docker (opcional)
docker build -t mlops-bc:latest .
docker run --rm -p 8000:8000 mlops-bc:latest

# 4) Tests
pytest -q

# 5) GitHub (dispara CI)
git add .
git commit -m "Entrega EM10"
git push
```
