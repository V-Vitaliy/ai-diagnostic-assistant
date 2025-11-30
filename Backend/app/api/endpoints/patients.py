from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List



from ...db.deps import get_db
from ...schemas.patient import PatientRead, PatientCreate
from ...services.patient_service import create_patient, get_patient, get_patients


router = APIRouter(
    prefix="/patients",
    tags=["Patients"]
)




@router.post("/", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def create_patient_endpoint(
    patient_data: PatientCreate,
    db: Session = Depends(get_db)
):

    db_patient = create_patient(db, patient_data)
    return db_patient


@router.get("/", response_model=List[PatientRead])
def get_patients_list_endpoint(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):

    patients = get_patients(db, skip=skip, limit=limit)
    return patients


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient_endpoint(
    patient_id: int,
    db: Session = Depends(get_db)
):

    db_patient = get_patient(db, patient_id)
    if db_patient is None:

        raise HTTPException(status_code=404, detail="Patient not found")
    return db_patient

