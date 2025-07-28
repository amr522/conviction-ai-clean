FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=staging

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "http.server", "8000"]
