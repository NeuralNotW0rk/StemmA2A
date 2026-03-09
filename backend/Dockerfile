# Use an official PyTorch image with Python 3.11 and CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Upgrade pip and setuptools (still a good practice)
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements and install
COPY requirements.txt .
# torch is already in the base image, so we don't need to install it.
# The requirements.txt file does not contain torch, so this is fine.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Set the default command
CMD ["python", "app.py"]