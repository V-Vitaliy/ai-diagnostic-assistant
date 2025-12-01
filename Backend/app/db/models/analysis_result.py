from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
import uuid

from app.db.base import Base

class AnalysisResult(Base):
    """Database model for 'analysis_results' table."""
    __tablename__ = 'analysis_results'

    # Primary Key is now UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign Key to Patient (Patient ID stays Integer based on your previous design)
    patient_id: Mapped[int] = mapped_column(ForeignKey('patients.id'))

    # Metadata
    analysis_type: Mapped[str] = mapped_column(String(100))
    symptoms_input: Mapped[str] = mapped_column(Text)

    # Storage Paths (No more Base64 blobs in DB)
    image_storage_path: Mapped[str] = mapped_column(Text)
    heatmap_storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Data
    raw_model_outputs: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    llm_report: Mapped[str] = mapped_column(Text)

    # Relationships
    patient: Mapped["Patient"] = relationship(back_populates="analysis_results")

    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="analysis_result",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"AnalysisResult(id={self.id}, type='{self.analysis_type}')"