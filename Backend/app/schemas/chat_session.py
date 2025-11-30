from pydantic import BaseModel, UUID4
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatSessionCreate(BaseModel):
    """Schema for creating a new chat session."""
    patient_id: int
    # analysis_id is optional (we can chat about patient generally)
    analysis_id: Optional[int] = None

class ChatSessionResponse(BaseModel):
    """Schema for returning session details."""
    session_id: UUID4
    patient_id: int
    history_json: List[Dict[str, Any]] = []
    patient_name: Optional[str] = None # Added for frontend convenience
    updated_at: datetime

    class Config:
        from_attributes = True