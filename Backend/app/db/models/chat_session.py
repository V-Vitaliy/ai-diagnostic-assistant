from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, ForeignKey, func, DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID
import uuid

from ..base import Base


class ChatSession(Base):

    __tablename__ = 'chat_sessions'

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    patient_id: Mapped[int] = mapped_column(ForeignKey('patients.id'))

    analysis_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey('analysis_results.id'),
        nullable=True
    )

    history_json: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB)
    state_json: Mapped[Dict[str, Any]] = mapped_column(JSONB)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now()
    )


    patient: Mapped["Patient"] = relationship(back_populates="chat_sessions")

    analysis_result: Mapped[Optional["AnalysisResult"]] = relationship(
        back_populates="chat_sessions"
    )

    def __repr__(self):

        return (
            f"ChatSession(id={self.session_id}, patient_id={self.patient_id}, "
            f"analysis_id={self.analysis_id})"
        )