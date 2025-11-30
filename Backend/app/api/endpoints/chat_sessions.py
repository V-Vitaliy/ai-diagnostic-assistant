from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import uuid


from ...db.deps import get_db
from ...schemas.chat_session import ChatSessionRead, ChatSessionCreate, ChatSessionBase
from ...services import chat_session_service


router = APIRouter(
    prefix="/chats",
    tags=["Chat Sessions"]
)




@router.post("/", response_model=ChatSessionRead, status_code=status.HTTP_201_CREATED)
def create_chat_session_endpoint(
    session_data: ChatSessionCreate,
    db: Session = Depends(get_db)
):

    db_session = chat_session_service.create_chat_session(db, session_data)
    return db_session


@router.get("/{session_id}", response_model=ChatSessionRead)
def get_session_by_uuid_endpoint(
    session_id: uuid.UUID,
    db: Session = Depends(get_db)
):

    db_session = chat_session_service.get_session_by_uuid(db, session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return db_session


@router.get("/patient/{patient_id}", response_model=List[ChatSessionRead])
def get_sessions_by_patient_endpoint(
    patient_id: int,
    db: Session = Depends(get_db)
):

    sessions = chat_session_service.get_sessions_by_patient(db, patient_id)
    return sessions


@router.put("/{session_id}", response_model=ChatSessionRead)
def update_chat_session_endpoint(
    session_id: uuid.UUID,
    session_update: ChatSessionBase,
    db: Session = Depends(get_db)
):

    updated_session = chat_session_service.update_chat_session(db, session_id, session_update)
    if updated_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return updated_session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat_session_endpoint(
    session_id: uuid.UUID,
    db: Session = Depends(get_db)
):

    deleted_session = chat_session_service.delete_chat_session(db, session_id)
    if deleted_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return