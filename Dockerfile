FROM python:3.12-slim


# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    bash \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Make script executable
RUN chmod +x run.sh

# Install Python deps (add wheel + setuptools + build for pyproject.toml builds)
RUN pip install --upgrade pip setuptools wheel build && \
    if [ -f "req.txt" ]; then pip install -r req.txt; fi
