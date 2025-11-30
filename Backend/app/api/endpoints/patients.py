from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.db.session import get_db
from app.db.models.patient import Patient
from app.schemas import PatientCreate, PatientResponse

router = APIRouter()

@router.post("/", response_model=PatientResponse)
async def create_patient(patient: PatientCreate, db: AsyncSession = Depends(get_db)):
    """
    Create a new patient in the database.
    """
    new_patient = Patient(
        name=patient.name,
        birth_date=patient.birth_date,
        chronic_diseases=patient.chronic_diseases,
        allergies=patient.allergies,
        medications=patient.medications,
        height_cm=patient.height_cm,
        weight_kg=patient.weight_kg
    )
    db.add(new_patient)
    await db.commit()
    await db.refresh(new_patient)
    return new_patient

@router.get("/", response_model=List[PatientResponse])
async def get_patients(db: AsyncSession = Depends(get_db)):
    """
    Get all patients.
    """
    result = await db.execute(select(Patient))
    return result.scalars().all()

@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get a specific patient by ID.
    """
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient