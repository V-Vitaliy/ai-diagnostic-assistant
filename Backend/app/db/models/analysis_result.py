from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey('patients.id'))

    analysis_type: Mapped[str] = mapped_column(String(100))
    image_storage_path: Mapped[str] = mapped_column(Text)
    symptoms_input: Mapped[str] = mapped_column(Text)

    raw_model_outputs: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    llm_report: Mapped[str] = mapped_column(Text)
    heatmap_base64: Mapped[str] = mapped_column(Text)

    patient: Mapped["Patient"] = relationship(back_populates="analysis_results")

    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="analysis_result",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"AnalysisResult(id={self.id}, patient_id={self.patient_id}, type='{self.analysis_type}')"