# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Set environment variable pointing to the trained model artifact
ENV MODEL_PATH=/app/models/random_forest_model.joblib

# Command to run the FastAPI model serving API
CMD ["uvicorn", "tests.test_api:app", "--host", "0.0.0.0", "--port", "8000"]