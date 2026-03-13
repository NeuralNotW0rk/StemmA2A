# 1. Use the slim Python 3.11 base
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install essential C++ build tools and OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install PyTorch targeting CUDA 13.0
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 5. Install the rest of your app's requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the code
COPY . .

CMD ["python", "engine_service.py"]