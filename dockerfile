# Imagen base ligera
FROM python:3.11-slim

# Evitar .pyc y salida bufferizada
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalar dependencias del sistema mínimas para pandas/numpy/yfinance
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar primero requirements (cache de capas)
COPY requirements.txt /app/

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY app.py /app/

# Exponer puerto (5001 en vez de 8000)
EXPOSE 5001

# Arrancar el servidor en 0.0.0.0:5001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
