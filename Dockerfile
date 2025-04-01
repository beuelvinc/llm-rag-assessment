FROM python:3.10-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    bash \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install latest Rust via rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    echo 'export PATH="/root/.cargo/bin:$PATH"' >> /root/.bashrc

WORKDIR /app

COPY . .
RUN chmod +x run.sh
RUN . /root/.cargo/env && \
    pip install --upgrade pip && \
    if [ -f "req.txt" ]; then pip install -r req.txt; fi

