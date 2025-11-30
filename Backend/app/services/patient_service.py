from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete


from ..db.models.patient import Patient
from ..schemas.patient import PatientCreate, PatientRead, PatientBase






def create_patient(db: Session, patient_data: PatientCreate) -> Patient:


    db_patient = Patient(**patient_data.model_dump())


    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)

    return db_patient




def get_patient(db: Session, patient_id: int) -> Optional[Patient]:



    stmt = select(Patient).where(Patient.id == patient_id)
    return db.scalar(stmt)


def get_patients(db: Session, skip: int = 0, limit: int = 100) -> List[Patient]:


    # stmt = select(Patient).offset(skip).limit(limit).order_by(Patient.id)
    # return db.scalars(stmt).all()

    # Простой способ:
    return db.query(Patient).offset(skip).limit(limit).all()




def update_patient(db: Session, patient_id: int, patient_update: PatientBase) -> Optional[Patient]:


    db_patient = get_patient(db, patient_id)
    if not db_patient:
        return None


    update_data = patient_update.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        setattr(db_patient, key, value)

    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)

    return db_patient



def delete_patient(db: Session, patient_id: int) -> Optional[Patient]:


    db_patient = get_patient(db, patient_id)

    if db_patient:
        db.delete(db_patient)
        db.commit()
        return db_patient

    return None