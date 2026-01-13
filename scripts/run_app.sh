#!/bin/bash
# App Orchestration Script

function cleanup {
    echo "[INFO] Stopping all services..."
    kill $(jobs -p)
}
trap cleanup EXIT

echo "[INFO] Starting Database Container..."
docker-compose up -d db

echo "[INFO] Waiting for database readiness..."
sleep 5

echo "[INFO] Starting Backend Service (FastAPI)..."
cd Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "[INFO] Starting Frontend Service (Streamlit)..."
cd ../Frontend
streamlit run app.py --server.port 8501 &
FRONTEND_PID=$!

echo "[INFO] System running. Press CTRL+C to stop."
wait $BACKEND_PID $FRONTEND_PID