# backend/app/api/endpoints/analysis.py (Version for Task B1)
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import Dict, Annotated
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/")
async def run_analysis(
    # --- Parameters without defaults first ---
    analysis_type: Annotated[str, Form(...)],
    patient_id: Annotated[int, Form(...)],
    image_file: Annotated[UploadFile, File(...)],
    # --- Parameters with defaults last ---
    symptoms: Annotated[str, Form()] = ""
) -> Dict:
    """
    Stub endpoint: Receives analysis data and image file, confirms reception.
    Does NOT call AI model yet. Task B1
    """
    logger.info(f"Received analysis request for patient {patient_id}. Type: {analysis_type}")

    image_contents = await image_file.read()
    if not image_contents:
        logger.error("Error: Image file is empty.")
        raise HTTPException(status_code=400, detail="Image file is empty.")
    logger.info(f"Image file '{image_file.filename}' received successfully ({len(image_contents)} bytes).")
    logger.info(f"Symptoms received: '{symptoms}'")

    return {
        "message": "Data received successfully by endpoint.",
        "patient_id": patient_id,
        "analysis_type": analysis_type,
        "filename": image_file.filename,
        "symptoms_received": symptoms
    }