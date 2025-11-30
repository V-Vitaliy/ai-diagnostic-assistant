import logging
from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from typing import AsyncGenerator



from .api.endpoints.analysis import router as analysis_router
from .api.endpoints.patients import router as patients_router
from .api.endpoints.chat_sessions import router as chat_sessions_router



from .db.deps import get_db as get_db_dependency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI Diagnostic Assistant API")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting FastAPI application...")
    logger.info("Application ready to accept requests.")
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title="AI Diagnostic Assistant API",
    description="API for analyzing medical data using AI models.",
    version="0.1.0",
    lifespan=lifespan
)


app.include_router(analysis_router, prefix="/analyze", tags=["Analysis Functions"])
app.include_router(patients_router, prefix="/patients", tags=["Patient Management"])
app.include_router(chat_sessions_router, prefix="/chats", tags=["Chat Sessions"])


@app.get("/", tags=["Root"])
def read_root():
    logger.info("Request to root endpoint /")
    return {"message": "Welcome to the AI Diagnostic Assistant API!"}


@app.get("/check-db-connection/", tags=["Testing"])

async def check_db_connection(db: AsyncSession = Depends(get_db_dependency)):

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