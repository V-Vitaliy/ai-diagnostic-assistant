from typing import List, Optional
import uuid  # Необходим для работы с UUID
from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete, func


from ..db.models.chat_session import ChatSession
from ..schemas.chat_session import ChatSessionCreate, ChatSessionBase






def create_chat_session(db: Session, session_data: ChatSessionCreate) -> ChatSession:


    db_session = ChatSession(**session_data.model_dump())

    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    return db_session




def get_session_by_uuid(db: Session, session_id: uuid.UUID) -> Optional[ChatSession]:


    stmt = select(ChatSession).where(ChatSession.session_id == session_id)
    return db.scalar(stmt)


def get_sessions_by_patient(db: Session, patient_id: int) -> List[ChatSession]:


    stmt = (
        select(ChatSession)
        .where(ChatSession.patient_id == patient_id)
        .order_by(ChatSession.updated_at.desc())
    )
    return list(db.scalars(stmt).all())




def update_chat_session(db: Session, session_id: uuid.UUID, session_update: ChatSessionBase) -> Optional[ChatSession]:


    db_session = get_session_by_uuid(db, session_id)
    if not db_session:
        return None

    update_data = session_update.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        setattr(db_session, key, value)

    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    return db_session




def delete_chat_session(db: Session, session_id: uuid.UUID) -> Optional[ChatSession]:
    """Удаляет сессию чата по UUID."""

    db_session = get_session_by_uuid(db, session_id)

    if db_session:
        db.delete(db_session)
        db.commit()
        return db_session

    return None