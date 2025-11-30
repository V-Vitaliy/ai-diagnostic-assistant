# backend/app/api/endpoints/__init__.py

from .patients import router as patients_router
from .analysis import router as analysis_router
from .chat_sessions import router as chat_sessions_router