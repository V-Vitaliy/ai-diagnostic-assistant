import logging
from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.api.endpoints import analysis, patients, chat

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI Diagnostic Assistant API")

app = FastAPI(
    title="AI Diagnostic Assistant API",
    description="API for analyzing medical data using AI models.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting FastAPI application...")
    logger.info("Application ready to accept requests.")

# --- Register Routers ---
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis Functions"])
app.include_router(patients.router, prefix="/patients", tags=["Patient Management"])
app.include_router(chat.router, prefix="/chat", tags=["AI Chat"])

@app.get("/", tags=["Root"])
def read_root():
    logger.info("Request to root endpoint /")
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