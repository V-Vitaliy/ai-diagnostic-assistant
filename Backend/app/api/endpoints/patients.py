from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.db.session import get_db
from app.db.models.patient import Patient
from app.db.models.user import User
from app.schemas import PatientCreate, PatientResponse
from app.api.deps import get_current_user

router = APIRouter()

@router.post("/", response_model=PatientResponse)
async def create_patient(
    patient: PatientCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new patient linked to the current logged-in user.
    """
    new_patient = Patient(
        name=patient.name,
        gender=patient.gender,
        birth_date=patient.birth_date,
        chronic_diseases=patient.chronic_diseases,
        allergies=patient.allergies,
        medications=patient.medications,
        height_cm=patient.height_cm,
        weight_kg=patient.weight_kg,
        user_id=current_user.id #
    )
    db.add(new_patient)
    await db.commit()
    await db.refresh(new_patient)
    return new_patient

@router.get("/", response_model=List[PatientResponse])
async def get_patients(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all patients belonging ONLY to the current user.
    """
    result = await db.execute(select(Patient).where(Patient.user_id == current_user.id))
    return result.scalars().all()

@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific patient (only if owned by user).
    """
    result = await db.execute(
        select(Patient).where(
            Patient.id == patient_id,
            Patient.user_id == current_user.id
        )
    )
    patient = result.scalar_one_or_none()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found or access denied")
    return patient