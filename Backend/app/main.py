import logging
from fastapi import FastAPI, Depends
from sqlalchemy import text

from app.api.endpoints import analysis, patients
from app.db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession

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


# Включаем роутеры
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis Functions"])
app.include_router(patients.router, prefix="/patients", tags=["Patient Management"])


@app.get("/", tags=["Root"])
def read_root():

    logger.info("Request to root endpoint /")
    return {"message": "Welcome to the AI Diagnostic Assistant API!"}



@app.get("/check-db-connection/", tags=["Testing"])
async def check_db_connection(db: AsyncSession = Depends(get_db)):

    try:

        result = await db.execute(text("SELECT 1"))
        one = result.scalar_one()

        return {
            "status": "success",
            "message": f"Successfully connected to DB and ran query. Result: {one}"
        }

    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return {
            "status": "error",
            "message": f"DB connection failed: {e}"
        }