# Optimized for RTX 40-series and CUDA 13.0 Hosts (Python 3.10)
FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app

# 1. Install system-level dependencies (essential for many Python AI libs)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade core Python build tools
# We use 'python -m pip' to ensure we're upgrading the correct instance
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Install your specific Python libraries
# REMINDER: Keep 'torch' OUT of requirements.txt to avoid massive redundant downloads
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your code
COPY . .

# 5. Security: Run as non-root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Standard API port for your app
EXPOSE 5000

CMD ["python", "app.py"]