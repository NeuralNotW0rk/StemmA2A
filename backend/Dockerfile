# 1. Start from NVIDIA's official CUDA developer image (standard Ubuntu 22.04 base)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Set the working directory
WORKDIR /app

# 3. Install Python 3.10, pip, and essential OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set python3.10 as the default 'python' and 'pip' commands
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install --no-cache-dir --upgrade pip wheel setuptools

# 5. Install PyTorch 2.5.1 with CUDA 12.1 support (using BuildKit cache mount to avoid re-downloading)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5b. Install packaging and ninja (prerequisites for flash-attn)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging ninja

# 5c. Install flash-attn
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flash-attn --no-build-isolation

# 6. Install the rest of the requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 7. Copy the code
COPY . .

# 8. Set up the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "engine_service.py"]