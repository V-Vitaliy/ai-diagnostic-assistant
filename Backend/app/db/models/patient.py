from typing import List, Optional
from datetime import date, datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Date, Integer, func, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base

class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    birth_date: Mapped[date] = mapped_column(Date, nullable=False)

    chronic_diseases: Mapped[List[str]] = mapped_column(JSONB, default=list)
    allergies: Mapped[List[str]] = mapped_column(JSONB, default=list)
    medications: Mapped[List[str]] = mapped_column(JSONB, default=list)

    height_cm: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    weight_kg: Mapped[Optional[int]] = mapped_column(Integer, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now()
    )

    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan"
    )

    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"Patient(id={self.id}, name='{self.name}', birth_date='{self.birth_date}')"