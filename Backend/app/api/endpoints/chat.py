from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
import uuid
import json

from app.db.session import get_db
from app.db.models.chat_session import ChatSession
from app.db.models.patient import Patient
from app.schemas.chat import ChatSessionCreate, ChatSessionResponse

# --- ГЛАВНОЕ ИЗМЕНЕНИЕ: Импортируем реальный сервис ---
# Убедитесь, что в llm_service.py исправлены импорты (from app.services...)
from app.services.llm_service import run_chat_workflow

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

        history_data = existing_session.history_json


        if isinstance(history_data, str):
            try:
                history_data = json.loads(history_data)
            except json.JSONDecodeError:
                history_data = []


        elif history_data is None:
            history_data = []


        return {
            "session_id": existing_session.session_id,
            "patient_id": patient.id,
            "history_json": history_data,
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
        import logging
        logging.error(f"AI Service Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")