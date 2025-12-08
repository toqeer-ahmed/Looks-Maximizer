# Use Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY backend/requirements.txt /app/requirements.txt

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the backend code and ml_pipeline
COPY backend /app/backend
COPY ml_pipeline /app/ml_pipeline

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Run the application - gunicorn with proper module path
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "120", "--workers", "2", "backend.app:app"]
