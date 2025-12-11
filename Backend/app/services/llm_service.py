import os
import json
import logging
from typing import List, Dict, Any, Optional
from functools import wraps
import asyncio

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

# --- FALLBACK CONFIGURATION ---
MODELS_PRIORITY = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-pro",
]
MAX_STORED_HISTORY = 40

# Global variable to track current working model
_current_model_index = 0
_model_failure_count = {}  # Track failures per model

# ==========================================
# FALLBACK SYSTEM
# ==========================================

class ModelFallbackError(Exception):
    """Raised when all models in fallback chain fail"""
    pass

def get_current_model() -> str:
    """Get the current active model"""
    global _current_model_index
    if _current_model_index >= len(MODELS_PRIORITY):
        _current_model_index = 0  # Reset to first model
    return MODELS_PRIORITY[_current_model_index]

def switch_to_next_model() -> Optional[str]:
    """Switch to next model in priority list"""
    global _current_model_index
    _current_model_index += 1
    if _current_model_index >= len(MODELS_PRIORITY):
        logger.error("All models exhausted in fallback chain")
        return None
    next_model = MODELS_PRIORITY[_current_model_index]
    logger.warning(f"Switching to fallback model: {next_model}")
    return next_model

def reset_model_selection():
    """Reset to the primary model (call after successful operation)"""
    global _current_model_index, _model_failure_count
    _current_model_index = 0
    _model_failure_count.clear()

def record_model_failure(model_name: str):
    """Track model failures"""
    global _model_failure_count
    _model_failure_count[model_name] = _model_failure_count.get(model_name, 0) + 1
    logger.warning(f"Model {model_name} failure count: {_model_failure_count[model_name]}")

def is_retryable_error(exception: Exception) -> bool:
    """Determine if error should trigger fallback"""
    error_msg = str(exception).lower()
    retryable_patterns = [
        "quota",
        "rate limit",
        "429",
        "503",
        "unavailable",
        "overloaded",
        "resource exhausted",
        "deadline exceeded",
        "timeout"
    ]
    return any(pattern in error_msg for pattern in retryable_patterns)

async def run_with_fallback(agent_func, *args, max_retries: int = None, **kwargs):
    """
    Run an agent function with automatic model fallback

    Args:
        agent_func: The async function to run
        max_retries: Maximum number of models to try (default: all models)
        *args, **kwargs: Arguments to pass to agent_func

    Returns:
        Result from successful execution

    Raises:
        ModelFallbackError: If all models fail
    """
    if max_retries is None:
        max_retries = len(MODELS_PRIORITY)

    last_exception = None
    attempts = 0

    while attempts < max_retries:
        current_model = get_current_model()
        logger.info(f"Attempt {attempts + 1}/{max_retries} using model: {current_model}")

        try:
            # Execute the function
            result = await agent_func(*args, **kwargs)

            # Success! Reset to primary model for next time
            if attempts > 0:
                logger.info(f"✅ Success with fallback model: {current_model}")
            reset_model_selection()
            return result

        except Exception as e:
            last_exception = e
            record_model_failure(current_model)
            logger.error(f"❌ Model {current_model} failed: {str(e)}")

            # Check if we should retry with next model
            if is_retryable_error(e):
                next_model = switch_to_next_model()
                if next_model is None:
                    break  # No more models to try

                attempts += 1
                await asyncio.sleep(1)  # Brief delay before retry
            else:
                # Non-retryable error, raise immediately
                logger.error(f"Non-retryable error encountered: {str(e)}")
                raise

    # All models failed
    error_msg = f"All {max_retries} models failed. Last error: {str(last_exception)}"
    logger.error(error_msg)
    raise ModelFallbackError(error_msg)


# -------------------------
# Helper serialization utils
# -------------------------
def serialize_history(contents: List[types.Content], limit: int = MAX_STORED_HISTORY) -> List[Dict[str, Any]]:
    """Convert a list of ADK types.Content -> JSON serializable list."""
    serialized = []
    tail = contents[-limit:] if len(contents) > limit else contents
    for content in tail:
        parts = []
        if hasattr(content, "parts") and content.parts is not None:
            for p in content.parts:
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
    """Try multiple ways to obtain a JSON-serializable state dict from runner.state."""
    state = getattr(runner, "state", None)
    if state is None:
        return {}
    try:
        dump = state.model_dump()
        return json.loads(json.dumps(dump, default=str))
    except Exception:
        pass
    try:
        dump = state.dict()
        return json.loads(json.dumps(dump, default=str))
    except Exception:
        pass
    try:
        if hasattr(state, "__dict__"):
            dump = {k: v for k, v in state.__dict__.items() if not k.startswith("_")}
            return json.loads(json.dumps(dump, default=str))
    except Exception:
        pass
    try:
        return json.loads(json.dumps(state, default=str))
    except Exception:
        logger.warning("Unable to fully serialize runner.state; saving empty state fallback.")
        return {}

def restore_state_safe(runner: InMemoryRunner, state_dict: Dict[str, Any]) -> None:
    """Try to restore runner.state from a dict using ADK-friendly constructors."""
    if not state_dict:
        return
    StateCls = getattr(runner, "State", None)
    if StateCls is not None:
        try:
            validated = StateCls.model_validate(state_dict)
            runner.state = validated
            return
        except Exception:
            pass
        try:
            if hasattr(StateCls, "parse_obj"):
                runner.state = StateCls.parse_obj(state_dict)
                return
        except Exception:
            pass
    try:
        runner.state = state_dict
    except Exception as e:
        logger.exception("Failed to attach restored state to runner: %s", e)

# ==========================================
# SESSION SERVICE (DB CONNECTION)
# ==========================================
class PostgresSessionService(SessionService):
    """Connects Google ADK Runners to PostgreSQL database."""
    def __init__(self, db: AsyncSession):
        self.db = db

    async def load(self, session_id: str) -> Session:
        query = select(ChatSession).where(ChatSession.session_id == session_id)
        result = await self.db.execute(query)
        db_session = result.scalar_one_or_none()

        if not db_session:
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
        """Save runner.state and MANUAL history into DB row for session_id."""
        query = select(ChatSession).where(ChatSession.session_id == session_id)
        result = await self.db.execute(query)
        db_session = result.scalar_one_or_none()

        if not db_session:
            logger.error("Attempt to save non-existent session %s", session_id)
            return

        serialized_history = serialize_history(manual_history, limit=MAX_STORED_HISTORY)
        db_session.history_json = json.dumps(serialized_history, ensure_ascii=False)

        state_dict = serialize_state_safe(runner)
        db_session.state_json = json.dumps(state_dict, ensure_ascii=False)

        await self.db.commit()

# ==========================================
# AGENT DEFINITIONS (with dynamic model)
# ==========================================
def create_research_agent() -> Agent:
    return Agent(
        name="AgentBadacz",
        model=get_current_model(),
        instruction="""
        # ROLA: Analityk Medyczny (Medical Research Specialist)
        
        Jesteś ekspertem ds. badań medycznych z dostępem do bazy wiedzy Google Search. 
        Twoja rola to weryfikacja i pogłębienie analizy AI poprzez wyszukiwanie aktualnych źródeł medycznych.
        
        ## PROCES PRACY:
        
        ### 1. ANALIZA WEJŚCIOWA
        - Otrzymujesz wyniki analizy AI z prawdopodobieństwami patologii (np. Pneumonia: 0.85, Atelectasis: 0.62)
        - Identyfikuj TOP 3-5 patologii z najwyższym prawdopodobieństwem (>0.5)
        - Zwróć uwagę na kombinacje chorób współistniejących
        
        ### 2. WYSZUKIWANIE WYTYCZNYCH
        Dla każdej zidentyfikowanej patologii znajdź:
        
        **A. Oficjalne guidelines:**
        - Wytyczne WHO, European/American medical societies
        - Standardy leczenia z ostatnich 3-5 lat
        - Kryteria diagnostyczne (np. kryteria Jones dla gorączki reumatycznej)
        
        **B. Objawy kliniczne:**
        - Charakterystyczne objawy radiologiczne
        - Typowy przebieg choroby
        - Czerwone flagi wymagające natychmiastowej interwencji
        
        **C. Diagnostyka różnicowa:**
        - Jakie inne choroby mogą dawać podobny obraz
        - Jak odróżnić między podobnymi patologiami
        
        ### 3. WERYFIKACJA FARMAKOLOGICZNA
        Sprawdź leki pacjenta w kontekście wykrytych patologii:
        
        **Zgodność terapii:**
        - Czy obecne leki są odpowiednie dla zdiagnozowanych schorzeń?
        - Czy brakuje standardowej terapii?
        - Czy są przeciwwskazania lub interakcje?
        
        **Przykład:**
        ```
        Pacjent: Pneumonia (0.87) + COPD w wywiadzie
        Leki: Salbutamol, Budesonide
        ✓ Zgodne z COPD
        ⚠ Brak antybiotyku dla pneumonii - wymaga konsultacji
        ```
        
        ### 4. KONTEKST PACJENTA
        Uwzględnij:
        - **Wiek:** (geriatria, pediatria mają inne standardy)
        - **Choroby współistniejące:** (np. cukrzyca + infekcja = gorsze rokowanie)
        - **Alergie:** (ograniczenia w wyborze antybiotyków)
        - **Stan ogólny:** (czy pacjent wymaga hospitalizacji?)
        
        ### 5. FORMAT WYJŚCIA
        
        Strukturyzuj odpowiedź jako JSON:
        ```json
        {
          "verified_pathologies": [
            {
              "name": "Pneumonia",
              "ai_probability": 0.87,
              "clinical_criteria": "Infiltraty w obrazie RTG + gorączka + kaszel produktywny",
              "guidelines_source": "European Respiratory Society 2023",
              "severity": "moderate",
              "requires_immediate_action": false
            }
          ],
          "differential_diagnosis": [
            "Tuberculosis (wykluczyć poprzez test QuantiFERON)",
            "Pulmonary embolism (D-dimer jeśli objawy)"
          ],
          "medication_assessment": {
            "appropriate": ["Salbutamol - zgodny z COPD"],
            "missing": ["Antybiotyk (Amoksycylina/Klawulanian - wg wytycznych)"],
            "contraindications": []
          },
          "red_flags": [
            "Saturacja O2 <90% - rozważ hospitalizację",
            "Gorączka >39°C przez >3 dni"
          ],
          "references": [
            "ERS Guidelines for CAP 2023",
            "UpToDate: Community-acquired pneumonia in adults"
          ]
        }
        ```
        
        ## WAŻNE ZASADY:
        
         ZAWSZE:
        - Podawaj źródła (konkretne nazwy guidelines, rok publikacji)
        - Używaj aktualnej wiedzy medycznej (nie starszej niż 5 lat)
        - Myśl o bezpieczeństwie pacjenta (red flags!)
        - Bądź precyzyjny w terminologii medycznej
        
         NIGDY:
        - Nie ignoruj niskiego prawdopodobieństwa jeśli to poważna choroba (np. 0.3 dla Pneumothorax to wciąż istotne!)
        - Nie zakładaj, że AI ma 100% rację
        - Nie pomijaj interakcji leków
        - Nie dawaj ostatecznej diagnozy (to rola lekarza)
        
        ## PRIORYTET:
        Bezpieczeństwo pacjenta > Dokładność AI > Kompletność raportu
        
        Jeśli masz wątpliwości - zaznacz to wyraźnie i zasugeruj dodatkowe badania.
        """,
        tools=[google_search],
    )

def create_writer_agent() -> Agent:
    return Agent(
        name="AgentLekarz",
        model=get_current_model(),
        instruction="""
        # ROLA: Lekarz Konsultant (Medical Writer & Advisor)
        
        Jesteś doświadczonym lekarzem, który na podstawie danych analitycznych tworzy 
        zrozumiałe, klinicznie użyteczne raporty dla lekarzy prowadzących i pacjentów.
        
        ## TWOJE ZADANIA:
        
        ### 1. SYNTEZA INFORMACJI
        Otrzymujesz od AgentaBadacza:
        - Zweryfikowane patologie z guidelines
        - Ocenę leków pacjenta
        - Red flags i różnicówkę
        
        Twoim zadaniem jest przekształcić to w **spójny raport medyczny**.
        
        ### 2. STRUKTURA ODPOWIEDZI
        
        #### A. DLA LEKARZA (Tryb profesjonalny)
        
        ```markdown
        ## STRESZCZENIE KLINICZNE
        
        **Pacjent:** [wiek]lat, [płeć], [główne schorzenia współistniejące]
        **Analiza z dnia:** [data]
        
        ### OCENA OBRAZOWANIA AI
        Na podstawie analizy obrazu radiologicznego zidentyfikowano:
        
        1. **Pneumonia (prawdopodobieństwo: 87%)**
           - Obraz RTG: Infiltraty w dolnym płacie prawym
           - Zgodność z kryteriami ERS 2023: TAK
           - Nasilenie: umiarkowane
           
        2. **Atelectasis (prawdopodobieństwo: 62%)**
           - Prawdopodobnie wtórne do pneumonii
           - Wymaga kontroli po leczeniu
        
        ### DIAGNOZA RÓŻNICOWA
        Do rozważenia:
        - Gruźlica płuc (wykluczyć: QuantiFERON-TB, посев plwociny)
        - Zatorowość płucna (mało prawdopodobna przy braku czynników ryzyka, D-dimer w razie wątpliwości)
        - Rak płuca (kontrolne CT za 6 tyg. po wyleczeniu)
        
        ### OCENA TERAPII
        
        **Obecne leki:**
        ✓ Salbutamol (wziewny) - właściwy dla COPD
        ✓ Budesonide (wziewny) - właściwy dla COPD
        
        **Zalecenia farmakologiczne:**
        ⚠️ BRAK antybiotykoterapii - wg. wytycznych ERS należy włączyć:
           - I wybór: Amoksycylina/Klawulanian 875/125mg 2x dz. (7-10 dni)
           - Alternatywa (alergia): Moksyfloksacyna 400mg 1x dz.
        
        **Monitorowanie:**
        - Kontrola RTG za 4-6 tygodni
        - Obserwacja temperatury, saturacji
        - Jeśli brak poprawy po 48-72h → zmiana antybiotyku
        
        ### RED FLAGS - WYMAGA UWAGI
        🚨 Hospitalizacja jeśli:
        - SpO2 <90% na powietrzu
        - Tachypnoe >30/min
        - Hipotensja <90/60 mmHg
        - Zaburzenia świadomości
        
        ### PIŚMIENNICTWO
        - European Respiratory Society Guidelines for CAP 2023
        - UpToDate: Community-acquired pneumonia - treatment
        ```
        
        #### B. DLA PACJENTA (Tryb uproszczony)
        
        ```markdown
        ## Wyjaśnienie wyników badania
        
        Dzień dobry,
        
        Na podstawie Pana/Pani zdjęcia rentgenowskiego oraz dodatkowych danych, 
        nasz system wykrył zmiany sugerujące **zapalenie płuc** (pneumonię).
        
        ### Co to oznacza?
        Zapalenie płuc to infekcja tkanki płucnej, która wymaga leczenia antybiotykiem.
        Objawy to zazwyczaj:
        - Kaszel (często z plwociną)
        - Gorączka
        - Duszność
        - Ból w klatce piersiowej
        
        ### Dlaczego system to wykrył?
        Na zdjęciu widoczne są **infiltraty** - czyli obszary zaciemnienia w płucu,
        które wskazują na obecność stanu zapalnego.
        
        ### Co dalej?
        
        **⚠️ WAŻNE - wymagana konsultacja z lekarzem!**
        
        Prawdopodobnie lekarz:
        1. Przepisze antybiotyk (najczęściej na 7-10 dni)
        2. Może zlecić dodatkowe badania krwi
        3. Będzie chciał zobaczyć Pana/Panią ponownie za kilka dni
        
        **Proszę skonsultować się z lekarzem jeśli:**
        - Gorączka powyżej 39°C nie spada po 2 dniach
        - Pojawia się trudność w oddychaniu
        - Kaszel z krwią
        - Nasilający się ból w klatce piersiowej
        
        ### Czy to coś poważnego?
        Zapalenie płuc jest poważnym schorzeniem, ALE:
        ✓ Jest uleczalne przy odpowiednim leczeniu
        ✓ Większość osób wraca do zdrowia w 2-3 tygodnie
        ✓ Kluczem jest wczesne rozpoczęcie antybiotyku
        
        ### Pana/Pani leki
        Obecnie przyjmuje Pan/Pani leki na astmę/COPD - to dobrze, proszę je kontynuować.
        Lekarz prawdopodobnie doda antybiotyk.
        
        Życzę szybkiego powrotu do zdrowia!
        ```
        
        ### 3. ZASADY KOMUNIKACJI
        
        **Język polski:**
        - Używaj poprawnej terminologii medycznej (dla lekarzy)
        - Unikaj żargonu (dla pacjentów)
        - Zawsze tłumacz skróty przy pierwszym użyciu
        
        **Ton:**
        - Profesjonalny ale ciepły
        - Empatyczny (rozumiesz, że ludzie się martwią)
        - Asertywny przy red flags
        
        **Struktura:**
        - Nagłówki i punktowanie dla czytelności
        - Najważniejsze informacje na początku
        - Emoji tylko dla red flags (🚨⚠️) i pozytywnych info (✓)
        
        ### 4. DOSTOSOWANIE DO KONTEKSTU
        
        **Wiek pacjenta:**
        - <18 lat → mów o "rodzicach/opiekunach"
        - 65+ lat → częstsze kontrole, ryzyko powikłań
        - Ciąża → inne leki, inne wytyczne
        
        **Choroby współistniejące:**
        - Cukrzyca → gorsze gojenie, ryzyko powikłań
        - Niewydolność serca → ostrożność z płynami
        - Astma/COPD → gorsze rokowanie przy infekcjach
        
        ### 5. CO ZAWSZE PAMIĘTAĆ
        
         MUSISZ:
        - Podkreślić konieczność konsultacji z lekarzem na żywo
        - Wymienić red flags wymagające natychmiastowej reakcji
        - Wyjaśnić DLACZEGO coś jest ważne (nie tylko CO)
        - Zakończyć pozytywnie (ale realistycznie)
        
         NIE WOLNO:
        - Stawiać ostatecznej diagnozy ("to jest pneumonia" → "zmiany sugerujące pneumonię")
        - Przepisywać leków (możesz powiedzieć "lekarz prawdopodobnie przepisze")
        - Bagatelizować objawów
        - Używać medycznego żargonu bez wyjaśnienia
        
        ### 6. FORMAT ODPOWIEDZI
        
        Automatycznie rozpoznaj tryb na podstawie kontekstu rozmowy:
        - Jeśli zapytanie od lekarza / medyczne szczegóły → tryb profesjonalny
        - Jeśli pytanie pacjenta / prośba o wyjaśnienie → tryb uproszczony
        - W razie wątpliwości → podaj obie wersje
        
        ## PAMIĘTAJ
        Twoja odpowiedź może bezpośrednio wpłynąć na decyzje kliniczne. 
        Zawsze priorytetuj bezpieczeństwo pacjenta nad "brzmieniem dobrze".
        """,
    )

# ==========================================
# WORKFLOW (ORCHESTRATOR WITH FALLBACK)
# ==========================================
async def _run_chat_workflow_internal(
        session_id: str,
        user_message: str,
        db: AsyncSession,
        patient: Patient,
        analysis_text: str,
        patient_text: str,
        current_session: Session
) -> str:
    """Internal function that actually runs the workflow (wrapped by fallback)"""

    root_instruction = f"""
    Jesteś Głównym Koordynatorem Medycznym.
    
    === DANE PACJENTA ===
    {patient_text}
    
    === HISTORIA BADAŃ (Ostatnie 5) ===
    {analysis_text}
    
    Twoje zadanie: Analizuj bieżące zapytanie użytkownika w kontekście całej historii badań pacjenta.
    Porównuj wyniki w czasie (np. czy stan się pogarsza).
    """

    # Initialize Sub-Agents with current model
    research_agent = create_research_agent()
    writer_agent = create_writer_agent()
    research_tool = AgentTool(agent=research_agent)
    writer_tool = AgentTool(agent=writer_agent)

    root_agent = Agent(
        name="Koordynator",
        model=get_current_model(),  # Dynamic model selection
        instruction=root_instruction,
        tools=[research_tool, writer_tool],
    )

    # Initialize Runner
    runner = InMemoryRunner(agent=root_agent)

    # Restore state
    if getattr(current_session, "state", None):
        try:
            restore_state_safe(runner, current_session.state)
        except Exception as e:
            logger.exception("Failed to restore runner.state: %s", e)

    # Manual History Tracking
    tracked_history = list(current_session.history)

    print(f"🤖 Agent running for {session_id} with model {get_current_model()}...")

    # Run agent
    response = await runner.run_debug(user_message)

    # Parse response
    final_text = ""
    try:
        if isinstance(response, str):
            final_text = response
        elif hasattr(response, "text") and response.text:
            final_text = response.text
        elif isinstance(response, list):
            for event in reversed(response):
                if hasattr(event, "content") and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            if event.content.role == "model":
                                final_text = part.text
                                break
                        if hasattr(part, "function_response") and part.function_response:
                            resp = part.function_response.response
                            if isinstance(resp, dict) and "result" in resp:
                                final_text = resp["result"]
                                break
                if final_text: break
        if not final_text:
            final_text = str(response)
    except Exception:
        final_text = str(response)

    # Update History
    tracked_history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
    tracked_history.append(types.Content(role="model", parts=[types.Part(text=final_text)]))

    # Save to DB
    session_service = PostgresSessionService(db)
    await session_service.save(session_id, runner, manual_history=tracked_history)

    return final_text


async def run_chat_workflow(
        session_id: str,
        user_message: str,
        db: AsyncSession
) -> str:
    """Main workflow entry point with fallback support"""

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

    # 3. Load analyses history
    analysis_text = "Brak dostępnych wyników analizy."
    analyses_query = (
        select(AnalysisResult)
        .where(AnalysisResult.patient_id == patient.id)
        .order_by(AnalysisResult.created_at.desc())
        .limit(5)
    )
    analyses_res = await db.execute(analyses_query)
    analyses_list = analyses_res.scalars().all()

    if analyses_list:
        analysis_text_blocks = []
        for idx, analysis in enumerate(analyses_list):
            formatted_findings = format_findings(analysis.raw_model_outputs)
            symptoms_text = analysis.symptoms_input or "Brak"
            date_str = analysis.created_at.strftime("%Y-%m-%d %H:%M") if analysis.created_at else "Nieznana data"
            block = f"""
            --- BADANIE #{idx+1} ({date_str}) ---
            ID: {analysis.id}
            Typ: {analysis.analysis_type}
            Objawy przy przyjęciu: "{symptoms_text}"
            Wyniki AI:
            {formatted_findings}
            """
            analysis_text_blocks.append(block)
        analysis_text = "\n".join(analysis_text_blocks)

    # 4. Format patient info
    age = None
    try:
        age = 2025 - patient.birth_date.year
    except:
        pass

    patient_text = format_patient_info({
        "age": age,
        "chronic_diseases": getattr(patient, "chronic_diseases", None),
        "allergies": getattr(patient, "allergies", None),
        "medications": getattr(patient, "medications", None)
    })

    # 5. Load session
    session_service = PostgresSessionService(db)
    current_session = await session_service.load(session_id)

    # 6. Run with fallback
    try:
        result = await run_with_fallback(
            _run_chat_workflow_internal,
            session_id=session_id,
            user_message=user_message,
            db=db,
            patient=patient,
            analysis_text=analysis_text,
            patient_text=patient_text,
            current_session=current_session,
            max_retries=len(MODELS_PRIORITY)
        )
        return result
    except ModelFallbackError as e:
        logger.error(f"All models failed for session {session_id}: {str(e)}")
        return f"Przepraszamy, wszystkie modele AI są obecnie niedostępne. Spróbuj ponownie za chwilę."
    except Exception as e:
        logger.exception("Unexpected error in chat workflow: %s", e)
        return f"Wystąpił błąd podczas generowania odpowiedzi: {str(e)}"