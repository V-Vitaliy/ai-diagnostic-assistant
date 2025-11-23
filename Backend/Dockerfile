# 1. Use the official Python image
FROM python:3.10-slim
WORKDIR /app

# 2. Install system dependencies
# We KEEP 'build-essential' and 'cmake' (for lightgbm)
# We KEEP the OpenCV dependencies (for pytorch-gradcam, cv2)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    # --- OpenCV Dependencies ---
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # --- End OpenCV Dependencies ---
    && rm -rf /var/lib/apt/lists/*

# 3. Copy ONLY requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# 5. Copy the rest of the app code
# (Will IGNORE model.pt because of .dockerignore)
COPY . .

# 6. Command to run the Uvicorn server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]