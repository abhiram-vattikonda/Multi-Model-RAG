FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsm6 \
    libxext6 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install compatible packaging tools (pkg_resources included)
RUN pip install --no-cache-dir \
    pip==24.0 \
    setuptools==70.0.0 \
    wheel==0.43.0

# Copy requirements file
COPY requirements.txt .

# Install Whisper separately without build isolation
RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

# Remove Whisper from requirements to avoid duplicate installation
RUN grep -v "openai-whisper" requirements.txt > requirements_no_whisper.txt

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements_no_whisper.txt

# Copy application code
COPY . .

EXPOSE 8000