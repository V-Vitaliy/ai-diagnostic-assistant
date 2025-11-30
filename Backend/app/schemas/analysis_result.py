from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AnalysisCreate(BaseModel):
    """
    Schema for initiating an analysis.
    Note: The image file is handled separately via UploadFile.
    """
    patient_id: int
    analysis_type: str
    symptoms_input: str

class AnalysisResponse(BaseModel):
    """Schema for returning analysis results to the frontend."""
    id: int
    patient_id: int
    analysis_type: str
    image_storage_path: str
    heatmap_base64: Optional[str] = None # Base64 string for frontend display
    llm_report: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True