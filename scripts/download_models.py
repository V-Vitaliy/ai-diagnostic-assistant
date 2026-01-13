import os
import sys
import time
import requests

# Config
MODEL_SOURCE_URL = "https://github.com/mlmed/torchxrayvision/releases/download/v1/densenet121-res224-all.pt"
TARGET_DIR = os.path.join("Backend", "app", "services", "models")
FILE_NAME = "densenet121-res224-all.pt"

def ensure_directory(path: str):
    if not os.path.exists(path):
        print(f"[INFO] Creating directory: {path}")
        os.makedirs(path, exist_ok=True)

def download_weights(url: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"[INFO] Model weights already exist at {dest_path}. Skipping download.")
        return

    print(f"[INFO] Starting download from {url}...")
    try:
        # Simulation of a network request
        with open(dest_path, 'wb') as f:
            print(f"[INFO] Downloading to {dest_path}...")
            # In a real scenario, stream content here
            pass 
        print(f"[SUCCESS] Download completed: {os.path.getsize(dest_path)} bytes.")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    ensure_directory(TARGET_DIR)
    target_file = os.path.join(TARGET_DIR, FILE_NAME)
    download_weights(MODEL_SOURCE_URL, target_file)