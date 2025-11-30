from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from pydantic import BaseModel, Field


class ChatSessionBase(BaseModel):

    # patient_id: INTEGER
    # Примечание: Pydantic требует здесь int, не 'patient_id: int'
    patient_id: int

    # analysis_id: INTEGER (Optional)
    analysis_id: Optional[int] = None
    # history_json: JSONB (List[Dict[str, Any]])
    history_json: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="История сообщений"
    )
    # state_json: JSONB (Dict[str, Any])
    state_json: Dict[str, Any] = Field(
        default_factory=dict,
        description="Состояние/память агента"
    )

class ChatSessionCreate(ChatSessionBase):

    pass

class ChatSessionRead(ChatSessionBase):
    # session_id: UUID
    session_id: uuid.UUID
    # updated_at: TIMESTAMPZ
    updated_at: datetime

    class Config:

        from_attributes = True