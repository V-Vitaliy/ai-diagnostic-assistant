from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AnalysisResultBase(BaseModel):


    analysis_type: str = Field(..., max_length=100)


    image_storage_path: str = Field(..., description="Путь к файлу изображения в хранилище")


    symptoms_input: str = Field(..., description="Симптомы, введенные врачом")


    raw_model_outputs: Dict[str, Any] = Field(..., description="Сырые данные JSON от модели AI")


    llm_report: str = Field(..., description="Конечный текстовый отчет от LLM")



    heatmap_base64: str = Field(..., description="Изображение карты тепла в Base64")

    patient_id: int = Field(..., description="ID пациента, к которому относится анализ")

class AnalysisResultCreate(AnalysisResultBase):
    pass

class AnalysisResultRead(AnalysisResultBase):

    id: int

    class Config:
        from_attributes = True  