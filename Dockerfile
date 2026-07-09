FROM python:3.11-slim

# tesseract-ocr : pytesseract (tools/ocr.py, agents/document_ocr_agent) ;
# fra/eng : langue OCR utilisée (tools/ocr.py, "fra+eng").
# libmagic1 : python-magic (tools/file_inspection.py) — fallback mimetypes
# sinon disponible mais dégradé.
# curl : healthcheck docker-compose sur /healthz.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fra \
        tesseract-ocr-eng \
        libmagic1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances installées avant la copie du code — le cache de layer Docker
# n'est invalidé que par un changement de requirements.txt, pas par chaque
# modification de code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
