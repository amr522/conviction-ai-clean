FROM python:3.10-slim

# Build arguments for versioning
ARG VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

# Labels for image metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="conviction-ai-pipeline" \
      org.label-schema.description="Conviction AI ETL and ML Training Pipeline" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-organization/conviction-ai-clean" \
      org.label-schema.schema-version="1.0"

WORKDIR /app

# Install system dependencies including AWS CLI and jq
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    jq \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Copy requirements and install Python dependencies
COPY requirements.txt dev-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r dev-requirements.txt

# Copy entire application
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh run_historical_pipeline.sh

# Create necessary directories
RUN mkdir -p data/Parquet_data staged master datasets logs

# Set default entrypoint to run full pipeline
ENTRYPOINT ["python", "src/run_full_pipeline.py"]
