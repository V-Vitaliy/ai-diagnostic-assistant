from pydantic import BaseModel, UUID4
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatSessionCreate(BaseModel):
    patient_id: int
    # Analysis ID is optional, but if provided, it must be UUID
    analysis_id: Optional[UUID4] = None

class ChatSessionResponse(BaseModel):
    session_id: UUID4
    patient_id: int
    history_json: List[Dict[str, Any]] = []
    patient_name: Optional[str] = None
    updated_at: datetime

    class Config:
        from_attributes = True