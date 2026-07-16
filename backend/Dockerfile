# 1. Start from NVIDIA's official CUDA developer image (standard Ubuntu 22.04 base)
ARG CUDA_VERSION=12.6.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

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
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Set python3.10 as the default 'python' and 'pip' commands
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install --no-cache-dir --upgrade pip wheel setuptools

# 5. Install PyTorch (using BuildKit cache mount to avoid re-downloading)
ARG TORCH_VERSION=2.7.1
ARG TORCH_CUDA_URL=https://download.pytorch.org/whl/cu126
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==${TORCH_VERSION} torchvision torchaudio --index-url ${TORCH_CUDA_URL}

# 5b. Install packaging and ninja (prerequisites for flash-attn)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging ninja

# 5c. Install flash-attn
# Target Turing (7.5), Ampere (8.0, 8.6), Ada (8.9), and Hopper (9.0) architectures.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ARG MAX_JOBS=2
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=${MAX_JOBS}
ENV FLASH_ATTENTION_FORCE_BUILD=1

RUN pip install flash-attn --no-build-isolation

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