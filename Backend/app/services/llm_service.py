import os
import json
import logging
from typing import List, Dict, Any, Optional

# --- Google API Key from Docker environment ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_GENAI_USE_VERTEXAI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not found in environment variables!")
else:
    logging.info("Google API Key loaded from environment")


# --- Google AI / ADK Imports ---
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search

# --- ADK Session Interface Handling (fallbacks if ADK SDK not present) ---
try:
    from google.adk.sessions import SessionService, Session
except Exception:
    class Session:
        def __init__(self, session_id: str, history: Optional[List[types.Content]] = None, state: Optional[Dict] = None):
            self.session_id = session_id
            self.history = history or []
            self.state = state or {}

    class SessionService:
        async def load(self, session_id: str) -> Session:
            raise NotImplementedError
        async def save(self, session_id: str, runner_state: Dict, history: List[Dict]) -> None:
            raise NotImplementedError

# --- Database Imports ---
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models import ChatSession, Patient, AnalysisResult

# --- Utility Imports ---
from app.services.prompt_builder import format_patient_info, format_findings


logger = logging.getLogger(__name__)
MODEL_NAME = "gemini-2.5-flash"
# max messages to keep in stored history
MAX_STORED_HISTORY = 40

# -------------------------
# Helper serialization utils
# -------------------------
def serialize_history(contents: List[types.Content], limit: int = MAX_STORED_HISTORY) -> List[Dict[str, Any]]:
    """
    Convert a list of ADK types.Content -> JSON serializable list.
    Keep only the last `limit` messages (most recent).
    """
    serialized = []
    tail = contents[-limit:] if len(contents) > limit else contents
    for content in tail:
        parts = []
        if hasattr(content, "parts") and content.parts is not None:
            for p in content.parts:
                # part may be a types.Part or plain dict or string
                text = None
                if hasattr(p, "text"):
                    text = getattr(p, "text")
                elif isinstance(p, dict) and "text" in p:
                    text = p["text"]
                elif isinstance(p, str):
                    text = p
                if text is not None:
                    parts.append({"text": text})
        serialized.append({
            "role": getattr(content, "role", None),
            "parts": parts
        })
    return serialized

def deserialize_history(raw_history: List[Dict[str, Any]]) -> List[types.Content]:
    restored = []
    for msg in raw_history:
        parts_objs = []
        for p in msg.get("parts", []):
            text = p.get("text") if isinstance(p, dict) else (p if isinstance(p, str) else None)
            if text is not None:
                parts_objs.append(types.Part(text=text))
        content = types.Content(role=msg.get("role"), parts=parts_objs)
        restored.append(content)
    return restored

def serialize_state_safe(runner: InMemoryRunner) -> Dict[str, Any]:
    """
    Try multiple ways to obtain a JSON-serializable state dict from runner.state.
    """
    state = getattr(runner, "state", None)
    if state is None:
        return {}
    # try model_dump (pydantic v2 / ADK)
    try:
        dump = state.model_dump()  # pydantic v2 style
        return json.loads(json.dumps(dump, default=str))
    except Exception:
        pass
    # try .dict()
    try:
        dump = state.dict()
        return json.loads(json.dumps(dump, default=str))
    except Exception:
        pass
    # try to convert attributes to dict
    try:
        if hasattr(state, "__dict__"):
            dump = {k: v for k, v in state.__dict__.items() if not k.startswith("_")}
            return json.loads(json.dumps(dump, default=str))
    except Exception:
        pass
    # fallback: try serializing basic fields
    try:
        return json.loads(json.dumps(state, default=str))
    except Exception:
        logger.warning("Unable to fully serialize runner.state; saving empty state fallback.")
        return {}

def restore_state_safe(runner: InMemoryRunner, state_dict: Dict[str, Any]) -> None:
    """
    Try to restore runner.state from a dict using ADK-friendly constructors.
    This will attempt several strategies; if none work, attach raw dict to runner.state.
    """
    if not state_dict:
        return
    # prefer runner.State.model_validate or similar if available
    StateCls = getattr(runner, "State", None)
    if StateCls is not None:
        # try model_validate (pydantic v2)
        try:
            validated = StateCls.model_validate(state_dict)
            runner.state = validated
            return
        except Exception:
            pass
        # try parse_obj or construct
        try:
            if hasattr(StateCls, "parse_obj"):
                runner.state = StateCls.parse_obj(state_dict)
                return
        except Exception:
            pass
    # fallback: attach raw dict
    try:
        runner.state = state_dict
    except Exception as e:
        logger.exception("Failed to attach restored state to runner: %s", e)

# ==========================================
# 1. SESSION SERVICE (DB CONNECTION)
# ==========================================
class PostgresSessionService(SessionService):
    """
    Connects Google ADK Runners to PostgreSQL database.
    Saves both: limited conversation history and serialized runner.state.
    """
    def __init__(self, db: AsyncSession):
        self.db = db

    async def load(self, session_id: str) -> Session:
        query = select(ChatSession).where(ChatSession.session_id == session_id)
        result = await self.db.execute(query)
        db_session = result.scalar_one_or_none()

        if not db_session:
            # return empty session; caller should ensure session existence if needed
            return Session(session_id=session_id, history=[], state={})

        try:
            raw_history = json.loads(db_session.history_json or "[]")
            restored_history = deserialize_history(raw_history)

            restored_state = json.loads(db_session.state_json or "{}")

            return Session(session_id=session_id, history=restored_history, state=restored_state)

        except Exception as e:
            logger.exception("Error loading session %s: %s", session_id, e)
            return Session(session_id=session_id, history=[], state={})

    async def save(self, session_id: str, runner: InMemoryRunner, manual_history: List[types.Content]) -> None:
        """
        Save runner.state and MANUAL history into DB row for session_id.
        NOTE: Since runner.history is missing, we must pass the tracked history manually.
        """
        query = select(ChatSession).where(ChatSession.session_id == session_id)
        result = await self.db.execute(query)
        db_session = result.scalar_one_or_none()

        if not db_session:
            logger.error("Attempt to save non-existent session %s", session_id)
            return

        # Serialize History (use the manually tracked history)
        serialized_history = serialize_history(manual_history, limit=MAX_STORED_HISTORY)
        db_session.history_json = json.dumps(serialized_history, ensure_ascii=False)

        # Serialize state
        state_dict = serialize_state_safe(runner)
        db_session.state_json = json.dumps(state_dict, ensure_ascii=False)

        await self.db.commit()

# ==========================================
# 2. AGENT DEFINITIONS (factory functions)
# ==========================================
def create_research_agent() -> Agent:
    return Agent(
        name="AgentBadacz",
        model=MODEL_NAME,
        instruction="""
        Role: Medical Researcher (Analityk Medyczny).
        Zadanie:
        1. Otrzymujesz wyniki analizy AI (prawdopodobieństwa chorób).
        2. Użyj Google Search, aby znaleźć wytyczne medyczne (guidelines) i objawy kliniczne dla wykrytych patologii o wysokim prawdopodobieństwie.
        3. Sprawdź, czy leki pacjenta są odpowiednie dla tych schorzeń.
        Wyjście: Tylko zweryfikowane fakty i źródła.
        """,
        tools=[google_search],
    )

def create_writer_agent() -> Agent:
    return Agent(
        name="AgentLekarz",
        model=MODEL_NAME,
        instruction="""
        Role: Medical Doctor (Lekarz).
        Zadanie:
        1. Na podstawie faktów od AgentaBadacza sformułuj diagnozę różnicową.
        2. Napisz zrozumiałą odpowiedź dla lekarza/pacjenta w języku polskim.
        3. Uwzględnij kontekst pacjenta (wiek, choroby współistniejące).
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
    # 1. Ensure ChatSession exists
    query = select(ChatSession).where(ChatSession.session_id == session_id)
    result = await db.execute(query)
    chat_db_obj = result.scalar_one_or_none()

    if not chat_db_obj:
        return "Błąd: Sesja nie istnieje."

    # 2. Get Patient Data
    pat_query = select(Patient).where(Patient.id == chat_db_obj.patient_id)
    pat_res = await db.execute(pat_query)
    patient = pat_res.scalar_one_or_none()
    if not patient:
        return "Błąd: Pacjent nie znaleziony."

    # --- FIX 1: LOAD ANALYSIS DATA (Inject into context) ---
    analysis_text = "Brak dostępnych wyników analizy."

     # Check if this session is linked to a specific analysis OR find the latest one for patient
    target_analysis_id = chat_db_obj.analysis_id

    if target_analysis_id:
        analysis_query = select(AnalysisResult).where(AnalysisResult.id == target_analysis_id)
    else:
        # Fallback: get latest analysis for patient based on CREATION DATE (Fixing random sort issue)
        analysis_query = select(AnalysisResult).where(AnalysisResult.patient_id == patient.id).order_by(AnalysisResult.created_at.desc())

    analysis_res = await db.execute(analysis_query)
    last_analysis = analysis_res.scalars().first()

    if last_analysis:
        formatted_results = format_findings(last_analysis.raw_model_outputs)
        symptoms_text = last_analysis.symptoms_input or "Brak zgłoszonych objawów"

        analysis_text = f"""
        ID Badania: {last_analysis.id}
        Typ: {last_analysis.analysis_type}

        === OBJAWY ZGŁOSZONE PRZEZ UŻYTKOWNIKA (SYMPTOMS) ===
        "{symptoms_text}"

       === SZCZEGÓŁOWE WYNIKI ANALIZY AI (FINDINGS) ===
        {formatted_results}

        """
    # 3. Format Context (Patient + Analysis)
    age = None
    try:
        age = 2025 - patient.birth_date.year
    except Exception:
        age = None

    patient_text = format_patient_info({
        "age": age,
        "chronic_diseases": getattr(patient, "chronic_diseases", None),
        "allergies": getattr(patient, "allergies", None),
        "medications": getattr(patient, "medications", None)
    })

    root_instruction = f"""
    Jesteś Głównym Koordynatorem Medycznym.

    === DANE PACJENTA ===
    {patient_text}

    === OSTATNIE WYNIKI BADAŃ (CONTEXT) ===
    {analysis_text}

    TWOJE ZADANIE:
    Koordynuj pracę agentów 'AgentBadacz' i 'AgentLekarz' w celu postawienia diagnozy.

    ZASADY DZIAŁANIA:
    Zarządzaj agentami Badacz i Lekarz. Jeśli użytkownik pyta o analizę, użyj danych z sekcji WYNIKI BADAŃ.
    Szczególną uwagę zwróć na sekcję "OBJAWY" - porównaj ją z wynikami AI.
    1. Jeśli w sekcji WYNIKI BADAŃ widzisz wysokie prawdopodobieństwo patologii (>50%), ZAWSZE wywołaj 'AgentBadacz', aby zweryfikował te wyniki w Google (szukaj wytycznych leczenia i objawów).
    2. Po weryfikacji, poproś 'AgentLekarz' o napisanie podsumowania klinicznego.
    3. Nie mów "nie mam danych". Dane są powyżej w sekcji CONTEXT.

    Język: Polski.
    """

    # 4. Load History & State from DB
    session_service = PostgresSessionService(db)
    current_session = await session_service.load(session_id)

    # 5. Initialize Sub-Agents
    research_agent = create_research_agent()
    writer_agent = create_writer_agent()
    research_tool = AgentTool(agent=research_agent)
    writer_tool = AgentTool(agent=writer_agent)

    root_agent = Agent(
        name="Koordynator",
        model=MODEL_NAME,
        instruction=root_instruction,
        tools=[research_tool, writer_tool],
    )

    # 6. Initialize Runner
    runner = InMemoryRunner(agent=root_agent)

    # Restore state
    if getattr(current_session, "state", None):
        try:
            restore_state_safe(runner, current_session.state)
        except Exception as e:
            logger.exception("Failed to restore runner.state: %s", e)

    # Manual History Tracking (since runner doesn't have .history)
    tracked_history = list(current_session.history)

    print(f"🤖 Agent running for {session_id}...")
    try:
        # Pass user message directly (since your version expects arg in run_debug)
        # Or append if using .run()

        # We assume your version of ADK's run_debug takes 'prompt' argument
        response = await runner.run_debug(user_message)

        # --- FIX 2: PARSE RESPONSE (Get clean text) ---
        final_text = ""
        try:
            # Check if response is string directly
            if isinstance(response, str):
                final_text = response
            # Check if response is standard ADK object with text attribute
            elif hasattr(response, "text") and response.text:
                final_text = response.text
            # Check if response is a LIST of events (common in ADK debug mode)
            elif isinstance(response, list):
                # Iterate backwards to find the final model output or tool output
                for event in reversed(response):
                    if hasattr(event, "content") and event.content.parts:
                        for part in event.content.parts:
                            # Check for plain text
                            if hasattr(part, "text") and part.text:
                                if event.content.role == "model":
                                    final_text = part.text
                                    break
                            # Check for function response (if agent didn't summarize)
                            if hasattr(part, "function_response") and part.function_response:
                                resp = part.function_response.response
                                if isinstance(resp, dict) and "result" in resp:
                                    final_text = resp["result"]
                                    break
                    if final_text: break

            # Fallback
            if not final_text:
                final_text = str(response)
        except Exception:
            final_text = str(response)

        # Update History Manually
        tracked_history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        tracked_history.append(types.Content(role="model", parts=[types.Part(text=final_text)]))

        # 8. Save updated history & state to DB
        await session_service.save(session_id, runner, manual_history=tracked_history)

        return final_text

    except Exception as e:
        logger.exception("LLM Error: %s", e)
        return f"Wystąpił błąd podczas generowania odpowiedzi: {str(e)}"