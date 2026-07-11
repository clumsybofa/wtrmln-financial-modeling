FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY wtrmln/ wtrmln/
COPY app.py watermelon-logo.png ./

# Session state, event log, and the encrypted vault live under /app/data —
# mount a volume here so connections survive restarts.
VOLUME ["/app/data"]

EXPOSE 8080
CMD ["uvicorn", "wtrmln.server:app", "--host", "0.0.0.0", "--port", "8080"]
