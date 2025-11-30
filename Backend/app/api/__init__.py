# backend/app/api/__init__.py

# Импортируем роутеры из подпакета endpoints (используем относительный импорт)
from .endpoints import patients_router, analysis_router, chat_sessions_router

# Экспортируем их под короткими именами, которые используются в main.py
analysis = analysis_router
patients = patients_router
chat_sessions = chat_sessions_router