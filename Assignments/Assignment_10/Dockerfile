# Minimal Dockerfile for Automotive MLOps Model Inference Service
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifact and inference script
COPY rf_model.joblib .
COPY predict.py .

# Run inference service entrypoint
CMD ["python", "predict.py"]
