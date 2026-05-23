FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias primero (capa cacheada)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación y recursos
COPY app.py .
COPY data/ data/
COPY models/ models/

EXPOSE 8050

CMD ["python", "app.py"]
