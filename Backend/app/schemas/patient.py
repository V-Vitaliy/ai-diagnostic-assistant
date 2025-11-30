from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime

class PatientBase(BaseModel):
    """
    Base properties for Patient.
    Used for creating and updating patient profiles.
    """
    name: str
    birth_date: date
    chronic_diseases: List[str] = []
    allergies: List[str] = []
    medications: List[str] = []
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None

class PatientCreate(PatientBase):
    """Schema for creating a patient (POST request)."""
    pass

class PatientResponse(PatientBase):
    """
    Schema for reading patient data (GET response).
    Includes DB-specific fields like ID and creation time.
    """
    id: int
    created_at: datetime

    class Config:
        from_attributes = True