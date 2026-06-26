# 1. Start from the official PyTorch 2.5.1 image (default Python 3.11)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# 2. Downgrade Python to 3.10 inside the conda environment and reinstall PyTorch for Python 3.10
RUN conda install -y python=3.10 && \
    pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    conda clean -ay

# 3. Set the working directory
WORKDIR /app

# 4. Install essential C++ build tools and OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Install the rest of the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the code
COPY . .

# 7. Set up the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "engine_service.py"]