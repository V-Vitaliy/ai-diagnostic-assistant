

from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete, func


from ..db.models.analysis_result import AnalysisResult
from ..schemas.analysis_result import AnalysisResultBase, AnalysisResultRead, AnalysisResultCreate

def create_analysis_result(db: Session, analysis_data: AnalysisResultCreate) -> AnalysisResult :
    db_analysis = AnalysisResult(**analysis_data.model_dump())
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis


def get_analysis_result(db: Session, analysis_id: int) -> Optional[AnalysisResult]:


    stmt = select(AnalysisResult).where(AnalysisResult.id == analysis_id)
    return db.scalar(stmt)


def get_patient_analysis_results(db: Session, patient_id: int) -> List[AnalysisResult]:


    stmt = select(AnalysisResult).where(AnalysisResult.patient_id == patient_id).order_by(AnalysisResult.id.desc())
    return list(db.scalars(stmt).all())




def update_analysis_result(db: Session, analysis_id: int, analysis_update: AnalysisResultBase) -> Optional[
    AnalysisResult]:


    db_analysis = get_analysis_result(db, analysis_id)
    if not db_analysis:
        return None


    update_data = analysis_update.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        setattr(db_analysis, key, value)

    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)

    return db_analysis




def delete_analysis_result(db: Session, analysis_id: int) -> Optional[AnalysisResult]:


    db_analysis = get_analysis_result(db, analysis_id)

    if db_analysis:
        db.delete(db_analysis)
        db.commit()
        return db_analysis

    return None