from typing import List, Optional
from datetime import date, datetime
import uuid
from pydantic import BaseModel, Field




class PatientBase(BaseModel):


    # name: VARCHAR(255), nullable=False
    name: str = Field(..., max_length=255, description="Имя и фамилия пациента")

    # birth_date: DATE, nullable=False
    birth_date: date = Field(..., description="Дата рождения пациента (YYYY-MM-DD)")

    # chronic_diseases: JSONB, default=list

    chronic_diseases: List[str] = Field(default_factory=list, description="Список хронических заболеваний")

    # allergies: JSONB, default=list
    allergies: List[str] = Field(default_factory=list, description="Список известных аллергий")

    # medications: JSONB, default=list
    medications: List[str] = Field(default_factory=list, description="Список постоянно принимаемых лекарств")

    # height_cm: INTEGER, default=None
    height_cm: Optional[int] = Field(None, ge=1, description="Рост в сантиметрах")

    # weight_kg: INTEGER, default=None
    weight_kg: Optional[int] = Field(None, ge=1, description="Вес в килограммах")




class PatientCreate(PatientBase):

    pass




class PatientRead(PatientBase):


    id: int = Field(..., description="Уникальный ID пациента")

    # created_at: TIMESTAMPZ
    created_at: datetime = Field(..., description="Дата создания профиля")


    class Config:

        from_attributes = True


