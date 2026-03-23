# backend/utils/filesystem.py
import os

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return dir
