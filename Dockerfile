FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by psycopg2, Pillow (OCR), and other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user for production security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# CMD is overridden per-service in docker-compose.prod.yml
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
