# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Evita bytecode y asegura stdout/stderr sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema (si fueran necesarias, aquí está el patrón)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entrenar automáticamente (opcional) si el modelo no existe:
RUN python -c "import os; os.path.exists('models/model.joblib') or __import__('train').train_and_save()"

EXPOSE 8000
ENV MODEL_PATH=models/model.joblib

# Iniciar con gunicorn (2 workers es suficiente para demo)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
