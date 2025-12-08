# Use Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY backend/requirements.txt /app/requirements.txt

# Install dependencies (including gunicorn)
# We also install libgl1 because opencv needs it
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the backend code and ml_pipeline
COPY backend /app/backend
COPY ml_pipeline /app/ml_pipeline

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$PYTHONPATH:/app \
    PORT=8000

# Expose port (Railway uses PORT env var)
EXPOSE $PORT

# Run the application - uses PORT env variable from Railway
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-8000} backend.app:app --timeout 120 --workers 2"]
