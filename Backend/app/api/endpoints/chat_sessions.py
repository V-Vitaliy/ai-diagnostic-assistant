from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
import uuid
import json
from datetime import datetime

# Импортируем подключение к БД
from app.db.session import get_db
# Импортируем модели
from app.db.models.chat_session import ChatSession
from app.db.models.patient import Patient
# Импортируем схемы (которые мы создали)
from app.schemas.chat import ChatSessionCreate, ChatSessionResponse

router = APIRouter()

class MessageRequest(BaseModel):
    session_id: str
    message: str

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
        return {
            "session_id": existing_session.session_id,
            "patient_id": patient.id,
            "history_json": existing_session.history_json or [],
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

@router.post("/message")
async def send_message(
    request: MessageRequest,
    db: AsyncSession = Depends(get_db)
):

    query = select(ChatSession).where(ChatSession.session_id == request.session_id)
    result = await db.execute(query)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    current_history = list(session.history_json) if session.history_json else []

    user_msg = {"role": "user", "parts": [{"text": request.message}]}
    current_history.append(user_msg)

    bot_msg = {
        "role": "model",
        "parts": [{"text": f"Система: Я получил ваше сообщение '{request.message}'. AI сервис пока отключен, но запись в БД работает!"}]
    }
    current_history.append(bot_msg)


    session.history_json = current_history
    session.updated_at = datetime.now()

    await db.commit()

    return {"response": bot_msg["parts"][0]["text"]}