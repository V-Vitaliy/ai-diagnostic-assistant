# backend/main.py
from fastapi import FastAPI

app = FastAPI(title="AI Diagnostic Assistant API")

@app.get("/")
def read_root():
    return {"message": " Backend dziala!"}