# Use official Python 3.9-slim runtime as base
FROM python:3.9-slim

# Set environment performance and configuration variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set container working directory
WORKDIR /app

# Install runtime dependencies for LightGBM (OpenMP) and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to optimize Docker layer caching
COPY requirements.txt .

# Upgrade pip and install pinned package requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend server code, static frontend assets and serialized model artifacts
COPY main.py .
COPY static/ static/
COPY Artifacts/ Artifacts/

# Expose target network port
EXPOSE 8000

# Run FastAPI server using production Uvicorn configuration
CMD ["uvicorn", "main:app", "--host", "0.0.5.0", "--host", "0.0.0.0", "--port", "8000"]
