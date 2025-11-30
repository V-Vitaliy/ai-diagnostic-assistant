import json
import logging
from typing import List, Dict, Any, Optional

from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search

# --- ВАЖНО: Импортируем базовый класс сессий из ADK (если он там есть) ---
# Если библиотека обновилась и там нет SessionService, мы просто создадим свой интерфейс.
# Но следуя твоей ссылке, мы пытаемся соответствовать структуре.
try:
    from google.adk.sessions import SessionService, Session
except ImportError:
    # Если в текущей версии SDK этого нет, создаем заглушки, чтобы код работал
    class Session:
        def __init__(self, session_id, history=None, state=None):
            self.session_id = session_id
            self.history = history or []
            self.state = state or {}
    class SessionService:
        pass

# --- Импорты Базы Данных ---
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models import ChatSession, Patient

# --- Утилиты ---
from ...app.services.prompt_builder import format_patient_info

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.0-flash-exp"

# ==========================================
# 1. СЕРВИС СЕССИЙ (ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ)
# ==========================================
class PostgresSessionService(SessionService):
    """
    Реализация SessionService для работы с PostgreSQL через SQLAlchemy.
    Это "мост", который объясняет Google ADK, как хранить данные в нашей БД.
    """
    def __init__(self, db: AsyncSession):
        self.db = db

    async def load(self, session_id: str) -> Session:
        """Загружает сессию из БД."""
        query = select(ChatSession).where(ChatSession.session_id == session_id)
        result = await self.db.execute(query)
        db_session = result.scalar_one_or_none()

        if not db_session:
            # Если сессии нет, возвращаем пустую
            return Session(session_id=session_id, history=[], state={})

        try:
            # Десериализация истории (JSON -> Objects)
            raw_history = json.loads(db_session.history_json or "[]")
            restored_history = []

            for msg in raw_history:
                # Восстанавливаем объекты Content
                # Внимание: тут нужно аккуратно мапить структуру под версию SDK
                parts = []
                if "parts" in msg:
                    for p in msg["parts"]:
                        if isinstance(p, dict) and "text" in p:
                            parts.append(types.Part(text=p["text"]))
                        elif isinstance(p, str): # Если вдруг сохранилось строкой
                             parts.append(types.Part(text=p))

                content = types.Content(role=msg["role"], parts=parts)
                restored_history.append(content)

            # Десериализация состояния (JSON -> Dict)
            restored_state = json.loads(db_session.state_json or "{}")

            return Session(
                session_id=session_id,
                history=restored_history,
                state=restored_state
            )

        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return Session(session_id=session_id, history=[], state={})

    async def save(self, session: Session):
        """Сохраняет сессию в БД."""
        query = select(ChatSession).where(ChatSession.session_id == session.session_id)
        result = await self.db.execute(query)
        db_session = result.scalar_one_or_none()

        if not db_session:
            # Если записи в БД нет, мы не можем сохранить (она должна создаваться через /init)
            logger.warning(f"Attempt to save non-existent session {session.session_id}")
            return

        # Сериализация (Objects -> JSON)
        serialized_history = []
        for content in session.history:
            parts_data = []
            for part in content.parts:
                # Сохраняем только текст, чтобы не ломать JSON бинарниками
                if part.text:
                    parts_data.append({"text": part.text})

            serialized_history.append({
                "role": content.role,
                "parts": parts_data
            })

        db_session.history_json = json.dumps(serialized_history, ensure_ascii=False)
        db_session.state_json = json.dumps(session.state, ensure_ascii=False)

        await self.db.commit()

# ==========================================
# 2. ФАБРИКИ АГЕНТОВ
# ==========================================
def create_research_agent() -> Agent:
    return Agent(
        name="AgentBadacz",
        model=MODEL_NAME,
        instruction="""
        Jesteś analitykiem medycznym (Medical Researcher). 
        Używaj Google Search do weryfikacji objawów i wyników AI.
        Szukaj sprzeczności i wytycznych medycznych.
        Zwracaj tylko fakty.
        """,
        tools=[google_search],
        output_key="research_findings",
    )

def create_writer_agent() -> Agent:
    return Agent(
        name="AgentLekarz",
        model=MODEL_NAME,
        instruction="""
        Jesteś lekarzem. Pisz profesjonalne raporty i odpowiedzi dla pacjenta/lekarza.
        Opieraj się na wynikach badań: {research_findings}.
        Język: Polski.
        """,
    )

# ==========================================
# 3. ORCHESTRATOR
# ==========================================
async def run_chat_workflow(
    session_id: str,
    user_message: str,
    db: AsyncSession
) -> str:
    # 1.
    query = select(ChatSession).where(ChatSession.session_id == session_id)
    result = await db.execute(query)
    chat_db_obj = result.scalar_one_or_none()

    if not chat_db_obj:
        return "Błąd: Sesja nie istnieje."

    pat_query = select(Patient).where(Patient.id == chat_db_obj.patient_id)
    pat_res = await db.execute(pat_query)
    patient = pat_res.scalar_one()

    patient_text = format_patient_info({
        "age": 2025 - patient.birth_date.year,
        "chronic_diseases": patient.chronic_diseases,
        "allergies": patient.allergies
    })

    # 2.
    root_instruction = f"""
    Jesteś Asystentem Medycznym.
    PACJENT: {patient_text}
    Zarządzaj agentami Badacz i Lekarz.
    """

    # 3.
    session_service = PostgresSessionService(db)

    # 4.
    current_session = await session_service.load(session_id)

    # 5.
    root_agent = Agent(
        name="Koordynator",
        model=MODEL_NAME,
        instruction=root_instruction,
        tools=[AgentTool(create_research_agent()), AgentTool(create_writer_agent())]
    )

    # 6.
    runner = InMemoryRunner(
        agent=root_agent,
        history=current_session.history,
        state=current_session.state
    )

    print(f"🤖 Agent running for {session_id}...")
    try:
        response = await runner.run(user_message)
        final_text = response.text
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return "Wystąpił błąd podczas generowania odpowiedzi."

    # 7.
    current_session.history = runner.history
    current_session.state = runner.state or {}

    await session_service.save(current_session)

    return final_text