import json
import logging
from typing import List, Dict, Any, Optional

# --- Google AI Imports ---
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search

# --- ADK Session Interface Handling ---
try:
    from google.adk.sessions import SessionService, Session
except ImportError:
    # Fallback if SDK version differs
    class Session:
        def __init__(self, session_id, history=None, state=None):
            self.session_id = session_id
            self.history = history or []
            self.state = state or {}
    class SessionService:
        pass

# --- Database Imports ---
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models import ChatSession, Patient

# --- Utility Imports ---
# Fixed the import path to be absolute
from app.services.prompt_builder import format_patient_info

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.0-flash-exp"

# ==========================================
# 1. SESSION SERVICE (DB CONNECTION)
# ==========================================
class PostgresSessionService(SessionService):
    """
    Connects Google ADK Runners to PostgreSQL database.
    """
    def __init__(self, db: AsyncSession):
        self.db = db

    async def load(self, session_id: str) -> Session:
        """Loads session history and state from DB."""
        query = select(ChatSession).where(ChatSession.session_id == session_id)
        result = await self.db.execute(query)
        db_session = result.scalar_one_or_none()

        if not db_session:
            return Session(session_id=session_id, history=[], state={})

        try:
            # Deserialize History
            raw_history = json.loads(db_session.history_json or "[]")
            restored_history = []

            for msg in raw_history:
                parts = []
                if "parts" in msg:
                    for p in msg["parts"]:
                        if isinstance(p, dict) and "text" in p:
                            parts.append(types.Part(text=p["text"]))
                        elif isinstance(p, str):
                             parts.append(types.Part(text=p))

                content = types.Content(role=msg["role"], parts=parts)
                restored_history.append(content)

            # Deserialize State
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
        """Saves session history and state to DB."""
        query = select(ChatSession).where(ChatSession.session_id == session.session_id)
        result = await self.db.execute(query)
        db_session = result.scalar_one_or_none()

        if not db_session:
            logger.warning(f"Attempt to save non-existent session {session.session_id}")
            return

        # Serialize History
        serialized_history = []
        for content in session.history:
            parts_data = []
            for part in content.parts:
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
# 2. AGENT DEFINITIONS
# ==========================================
def create_research_agent() -> Agent:
    return Agent(
        name="AgentBadacz",
        model=MODEL_NAME,
        instruction="""
        Role: Medical Researcher.
        Task: Verify symptoms and findings using Google Search.
        Output: Fact-based verification only.
        """,
        tools=[google_search],
        output_key="research_findings",
    )

def create_writer_agent() -> Agent:
    return Agent(
        name="AgentLekarz",
        model=MODEL_NAME,
        instruction="""
        Role: Medical Doctor.
        Task: Provide professional response in Polish based on research.
        """,
    )

# ==========================================
# 3. WORKFLOW (ORCHESTRATOR)
# ==========================================
async def run_chat_workflow(
    session_id: str,
    user_message: str,
    db: AsyncSession
) -> str:
    # 1. Get Chat Session
    query = select(ChatSession).where(ChatSession.session_id == session_id)
    result = await db.execute(query)
    chat_db_obj = result.scalar_one_or_none()

    if not chat_db_obj:
        return "Błąd: Sesja nie istnieje."

    # 2. Get Patient Data
    pat_query = select(Patient).where(Patient.id == chat_db_obj.patient_id)
    pat_res = await db.execute(pat_query)
    patient = pat_res.scalar_one()

    # 3. Format Context (INCLUDING MEDICATIONS)
    patient_text = format_patient_info({
        "age": 2025 - patient.birth_date.year,
        "chronic_diseases": patient.chronic_diseases,
        "allergies": patient.allergies,
        "medications": patient.medications # <--- Added this field!
    })

    root_instruction = f"""
    Jesteś Asystentem Medycznym.
    PACJENT: {patient_text}
    """

    # 4. Load History & State
    session_service = PostgresSessionService(db)
    current_session = await session_service.load(session_id)

    # 5. Initialize Root Agent
    root_agent = Agent(
        name="Koordynator",
        model=MODEL_NAME,
        instruction=root_instruction,
        tools=[AgentTool(create_research_agent()), AgentTool(create_writer_agent())]
    )

    # 6. Run
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

    # 7. Save
    current_session.history = runner.history
    current_session.state = runner.state or {}

    await session_service.save(current_session)

    return final_text