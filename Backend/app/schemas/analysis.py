from pydantic import BaseModel, UUID4
from typing import Optional, Dict, Any
from datetime import datetime

class AnalysisCreate(BaseModel):
    """Schema for initiating an analysis."""
    patient_id: int
    analysis_type: str
    symptoms_input: str

class AnalysisResponse(BaseModel):
    """Schema for returning analysis results."""
    # Updated ID type to UUID4
    id: UUID4
    patient_id: int
    analysis_type: str

    # Returning paths instead of blobs
    image_storage_path: str
    heatmap_storage_path: Optional[str] = None

    llm_report: Optional[str] = None
    image_analysis_results: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True