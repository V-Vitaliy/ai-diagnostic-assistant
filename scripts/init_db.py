import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
from app.db.base import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initialize_database():
    """
    Connects to the PostgreSQL instance and applies the schema.
    Warning: This might drop existing tables if configured to do so.
    """
    logger.info(f"Connecting to database at {settings.DATABASE_URL}...")
    
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        logger.info("Applying database schema...")
        # In production, use Alembic. For local dev, create_all is acceptable.
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database schema initialized successfully.")
    await engine.dispose()

if __name__ == "__main__":
    try:
        asyncio.run(initialize_database())
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        exit(1)