from fastapi import FastAPI
import logging
from .api.endpoints import analysis, patients


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Diagnostic Assistant API",
    description="API for analyzing medical data using AI models.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting FastAPI application...")
    logger.info("Application ready to accept requests.")


app.include_router(analysis.router, prefix="/analyze", tags=["Analysis Functions"])

app.include_router(patients.router, prefix="/patients", tags=["Patient Management"])


@app.get("/", tags=["Root"])
def read_root():
    """Returns a welcome message."""
    logger.info("Request to root endpoint /")
    return {"message": "Welcome to the AI Diagnostic Assistant API!"}