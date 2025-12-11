from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
import uuid
import json
import logging

from app.db.session import get_db
from app.db.models.chat_session import ChatSession
from app.db.models.patient import Patient
from app.schemas.chat import ChatSessionCreate, ChatSessionResponse

# --- ГЛАВНОЕ ИЗМЕНЕНИЕ: Импортируем реальный сервис ---
from app.services.llm_service import run_chat_workflow

logger = logging.getLogger(__name__)

router = APIRouter()

class MessageRequest(BaseModel):
    session_id: str
    message: str

class MessageResponse(BaseModel):
    response: str

@router.post("/session/init", response_model=ChatSessionResponse)
async def init_chat_session(
    request: ChatSessionCreate,
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(select(Patient).where(Patient.id == request.patient_id))
    patient = result.scalar_one_or_none()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    query = (
        select(ChatSession)
        .where(ChatSession.patient_id == request.patient_id)
        .order_by(ChatSession.updated_at.desc())
    )
    result = await db.execute(query)
    existing_session = result.scalar_one_or_none()

    if existing_session:
        # --- FIX: STRICT TYPE CHECKING & PARSING ---
        raw_history = existing_session.history_json
        final_history = []

        # Логируем тип данных для отладки
        logger.info(f"Session {existing_session.session_id} raw_history type: {type(raw_history)}")

        if raw_history is not None:
            # Случай 1: SQLAlchemy уже вернула список (для JSON полей)
            if isinstance(raw_history, list):
                final_history = raw_history

            # Случай 2: Пришла строка или байты (Text поле или сырой JSON)
            elif isinstance(raw_history, (str, bytes)):
                try:
                    parsed = json.loads(raw_history)

                    # Защита от двойного кодирования (строка внутри строки: "[...]")
                    if isinstance(parsed, str):
                        parsed = json.loads(parsed)

                    if isinstance(parsed, list):
                        final_history = parsed
                    else:
                        logger.warning(f"Parsed history is not a list, got: {type(parsed)}")
                        final_history = []
                except Exception as e:
                    logger.error(f"Error parsing history_json string: {e}")
                    final_history = []

            # Случай 3: Неизвестный тип
            else:
                logger.warning(f"Unexpected history_json type: {type(raw_history)}")
                final_history = []

        return {
            "session_id": existing_session.session_id,
            "patient_id": patient.id,
            "history_json": final_history,
            "patient_name": patient.name,
            "updated_at": existing_session.updated_at
        }
    else:
        new_session_id = uuid.uuid4()
        new_session = ChatSession(
            session_id=new_session_id,
            patient_id=request.patient_id,
            analysis_id=request.analysis_id,
            history_json=[],
            state_json={}
        )
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)

        return {
            "session_id": new_session_id,
            "patient_id": patient.id,
            "history_json": [],
            "patient_name": patient.name,
            "updated_at": new_session.updated_at
        }

@router.post("/message", response_model=MessageResponse)
async def send_message(
    request: MessageRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Sends the massage to Google Gemini.
    """
    try:
        ai_response_text = await run_chat_workflow(
            session_id=request.session_id,
            user_message=request.message,
            db=db
        )

        return {"response": ai_response_text}

    except Exception as e:
        logger.error(f"AI Service Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")