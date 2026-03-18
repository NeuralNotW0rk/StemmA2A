# 1. This image comes with Python 3.11 and Torch PRE-INSTALLED
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# 2. Set the working directory
WORKDIR /app

# 3. Install essential C++ build tools and OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install the rest of the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the code
COPY . .

CMD ["python", "engine_service.py"]