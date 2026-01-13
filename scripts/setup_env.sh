#!/bin/bash
# Env Setup Automation Script

set -e

echo "[INFO] Initializing Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[INFO] Upgrading pip tooling..."
pip install --upgrade pip setuptools wheel

echo "[INFO] Installing Backend dependencies..."
if [ -f "Backend/requirements.txt" ]; then
    pip install -r Backend/requirements.txt
else
    echo "[WARN] Backend requirements not found."
fi

echo "[INFO] Installing Frontend dependencies..."
if [ -f "Frontend/requirements.txt" ]; then
    pip install -r Frontend/requirements.txt
else
    echo "[WARN] Frontend requirements not found."
fi

echo "[SUCCESS] Environment setup completed successfully."