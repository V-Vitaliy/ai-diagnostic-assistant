from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Text, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
import uuid

from app.db.base import Base

class AnalysisResult(Base):
    """Database model for 'analysis_results' table."""
    __tablename__ = 'analysis_results'

    # Primary Key is UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    patient_id: Mapped[int] = mapped_column(ForeignKey('patients.id'))

    analysis_type: Mapped[str] = mapped_column(String(100))
    symptoms_input: Mapped[str] = mapped_column(Text)

    # Values: "PENDING", "COMPLETED", "FAILED"
    status: Mapped[str] = mapped_column(
        String(20),
        default="PENDING",
        server_default="PENDING",
        nullable=False
    )

    image_storage_path: Mapped[str] = mapped_column(Text)
    heatmap_storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Use default_factory to ensure new objects have empty dict, not None
    raw_model_outputs: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    # --- Timestamp ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    patient: Mapped["Patient"] = relationship(back_populates="analysis_results")

    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="analysis_result",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"AnalysisResult(id={self.id}, type='{self.analysis_type}', status='{self.status}')"