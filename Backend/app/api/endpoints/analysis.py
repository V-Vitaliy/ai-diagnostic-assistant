from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import Dict, Annotated, Any
import logging
import shutil
import os
import uuid

# --- Database Imports ---
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.db.models.analysis_result import AnalysisResult
from app.db.models.patient import Patient

# --- Import ALL AI service functions ---
# Ensure these imports match your project structure
from app.services.image_analysis import (
    analyze_chest_xray,
    analyze_extremity_xray,
    analyze_whole_body_ct_3d
)
from app.services.ocr_service import analyze_blood_image

logger = logging.getLogger(__name__)

router = APIRouter()

# Local directory to save uploaded images (Crucial for DB and Chat history)
UPLOAD_DIR = "app/static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def run_analysis(
    analysis_type: Annotated[str, Form(...)],
    patient_id: Annotated[int, Form(...)],
    image_file: Annotated[UploadFile, File(...)],
    symptoms: Annotated[str, Form()] = "",
    db: AsyncSession = Depends(get_db) # <--- Injected DB Session
) -> Dict[str, Any]:
    """
    1. Receives data.
    2. Validates Patient in DB.
    3. Saves file permanently (for Chat history).
    4. Routes to the appropriate AI model.
    5. Saves result to Database.
    """
    logger.info(f"Received analysis request for patient {patient_id}. Type: {analysis_type}")

    # --- 1. Validate Patient (Database Check) ---
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Patient not found")

    # --- 2. Save file to disk PERMANENTLY ---
    # We need a permanent path for the database, so the Chat Agent can reference it later.
    original_ext = image_file.filename.split(".")[-1] if "." in image_file.filename else "png"
    unique_filename = f"{uuid.uuid4()}.{original_ext}"
    file_path = f"{UPLOAD_DIR}/{unique_filename}"

    # Read file content
    file_content = await image_file.read()

    if not file_content:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    # Write to disk
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    logger.info(f"File saved permanently to: {file_path}")

    # --- 3. Run AI Analysis ---
    image_analysis_output = {}
    llm_report = "Report generation is not yet implemented." # Placeholder

    try:
        # Logic for 2D PNG/JPG files
        if analysis_type in ["chest_xray", "extremity_xray", "ocr"]:

            if analysis_type == "chest_xray":
                logger.info("Routing to chest x-ray analysis service...")
                image_analysis_output = analyze_chest_xray(image_bytes=file_content)

            elif analysis_type == "extremity_xray":
                logger.info("Routing to extremity (fracture) analysis service...")
                image_analysis_output = analyze_extremity_xray(image_bytes=file_content)

            elif analysis_type == "ocr":
                logger.info("Routing to OCR blood analysis service...")
                image_analysis_output = analyze_blood_image(image_bytes=file_content)

            logger.info(f"2D Image analysis completed successfully.")

        # Logic for 3D .nii.gz files
        elif analysis_type == "whole_body_ct":
            if not image_file.filename.endswith(('.nii', '.nii.gz')):
                raise HTTPException(status_code=400, detail="Invalid file type for whole_body_ct.")

            logger.info("Routing to 3D Whole Body CT analysis service...")

            # NOTE: We pass the PERMANENT file path now.
            image_analysis_output = analyze_whole_body_ct_3d(temp_file_path=file_path)

            logger.info(f"3D Image analysis (MONAI) completed successfully.")

        else:
            logger.warning(f"Analysis type '{analysis_type}' is not supported yet.")
            raise HTTPException(status_code=400, detail=f"Analysis type '{analysis_type}' is not supported.")

    except HTTPException as he:
         logger.error(f"Error relayed from AI service: {he.detail}")
         raise he
    except Exception as e:
        logger.exception(f"Unexpected error during AI analysis: {e}")
        # We catch error but proceed to save the attempt to DB as failed/empty
        image_analysis_output = {"error": str(e)}

    # --- 4. Save to Database (ENABLED) ---
    # Now we save the result so the Chat Agent can see it later.

    # Extract inner results if nested
    final_output_data = image_analysis_output.get("analysis_results", image_analysis_output)

    new_analysis = AnalysisResult(
        patient_id=patient_id,
        analysis_type=analysis_type,
        symptoms_input=symptoms,
        image_storage_path=file_path, # Path to the saved file
        raw_model_outputs=final_output_data,
        llm_report=llm_report,
        heatmap_base64="" # You can map this if your AI returns a heatmap string
    )

    db.add(new_analysis)
    await db.commit()
    await db.refresh(new_analysis)

    logger.info(f"Analysis result saved to Database with ID: {new_analysis.id}")

    # --- 5. Construct Response ---
    final_result = {
        "id": new_analysis.id, # Returning DB ID is crucial for the Frontend
        "patient_id": patient_id,
        "analysis_type": analysis_type,
        "image_analysis_results": final_output_data,
        "llm_report": llm_report
    }

    return final_result