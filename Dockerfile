FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Required at runtime — injected via HF Space secrets or docker run -e
ENV API_BASE_URL=""
ENV MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct"
ENV HF_TOKEN=""

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

