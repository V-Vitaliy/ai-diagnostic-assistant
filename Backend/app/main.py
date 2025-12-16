import logging
import os
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.api.endpoints import analysis, patients, chat, auth
from app.core.config import settings

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI Diagnostic Assistant API")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0"
)

# --- CORS CONFIGURATION ---
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting FastAPI application...")
    os.makedirs("app/static", exist_ok=True)
    logger.info("Application ready to accept requests.")

# Mount Static Files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Register Routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis Functions"])
app.include_router(patients.router, prefix="/patients", tags=["Patient Management"])
app.include_router(chat.router, prefix="/chat", tags=["AI Chat"])
app.include_router(auth.router, prefix="/auth", tags=["User Management"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the AI Diagnostic Assistant API!"}

@app.get("/check-db-connection/", tags=["Testing"])
async def check_db_connection(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("SELECT 1"))
        one = result.scalar_one()
        return {"status": "success", "message": f"Connected to DB. Result: {one}"}
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return {"status": "error", "message": str(e)}