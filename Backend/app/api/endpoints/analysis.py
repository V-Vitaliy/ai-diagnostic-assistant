from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import Dict, Annotated, Any
import logging
import shutil
import os
import uuid
import base64

# --- Database Imports ---
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.db.models.analysis_result import AnalysisResult
from app.db.models.patient import Patient

# --- AI Service Imports ---
from app.services.image_analysis import (
    analyze_chest_xray,
    analyze_extremity_xray,
    analyze_whole_body_ct_3d
)
from app.services.ocr_service import analyze_blood_image

logger = logging.getLogger(__name__)

router = APIRouter()

# Directories
UPLOAD_DIR = "app/static/uploads"
HEATMAP_DIR = "app/static/heatmaps"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

@router.post("/")
async def run_analysis(
    analysis_type: Annotated[str, Form(...)],
    patient_id: Annotated[int, Form(...)],
    image_file: Annotated[UploadFile, File(...)],
    symptoms: Annotated[str, Form()] = "",
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:

    logger.info(f"Start Analysis: Patient {patient_id}, Type {analysis_type}")

    # 1. Validate Patient
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Patient not found")

    # 2. Generate UUID for the analysis (We use this for filenames too)
    analysis_uuid = uuid.uuid4()

    # 3. Save Source File
    original_ext = image_file.filename.split(".")[-1] if "." in image_file.filename else "png"
    # Ensure source 3D files are saved with correct extension for MONAI
    if analysis_type == "whole_body_ct" and not original_ext.endswith("gz"):
        # Simple check, usually .nii.gz is handled as one
        pass
    # Use the analysis UUID for the filename
    source_filename = f"{analysis_uuid}.{original_ext}"
    file_path = f"{UPLOAD_DIR}/{source_filename}"

    file_content = await image_file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Empty file")

    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    # 4. Run AI Analysis
    image_analysis_output = {}

    try:
        if analysis_type == "chest_xray":
            image_analysis_output = analyze_chest_xray(image_bytes=file_content)
        elif analysis_type == "extremity_xray":
            image_analysis_output = analyze_extremity_xray(image_bytes=file_content)
        elif analysis_type == "ocr":
            image_analysis_output = analyze_blood_image(image_bytes=file_content)
        elif analysis_type == "whole_body_ct":
            image_analysis_output = analyze_whole_body_ct_3d(temp_file_path=file_path)
        else:
            image_analysis_output = {"error": "Type not supported"}

    except Exception as e:
        logger.error(f"AI Service Failed: {e}")
        image_analysis_output = {"error": str(e), "status": "failed"}

    # 5. Handle Heatmap (Save to Disk)
    # Extract inner results
    final_data = image_analysis_output.get("analysis_results", image_analysis_output)

    heatmap_base64 = final_data.get("heatmap_base64", "")
    heatmap_storage_path = None

    if heatmap_base64:
        try:
            heatmap_bytes = base64.b64decode(heatmap_base64)
            # Filename: uuid_heatmap.png
            heatmap_filename = f"{analysis_uuid}_heatmap.png"
            heatmap_full_path = f"{HEATMAP_DIR}/{heatmap_filename}"

            with open(heatmap_full_path, "wb") as f:
                f.write(heatmap_bytes)

            heatmap_storage_path = heatmap_full_path
            logger.info(f"Heatmap saved to: {heatmap_storage_path}")

            # Clean up JSON
            if "heatmap_base64" in final_data:
                del final_data["heatmap_base64"]

        except Exception as e:
            logger.error(f"Failed to save heatmap image: {e}")

    # 6. Save to Database
    new_analysis = AnalysisResult(
        id=analysis_uuid, # Explicitly setting UUID
        patient_id=patient_id,
        analysis_type=analysis_type,
        symptoms_input=symptoms,
        image_storage_path=file_path,
        heatmap_storage_path=heatmap_storage_path, # Path instead of blob
        raw_model_outputs=final_data,
    )

    db.add(new_analysis)
    await db.commit()
    await db.refresh(new_analysis)

    return {
        "id": new_analysis.id,
        "patient_id": patient_id,
        "analysis_type": analysis_type,
        "image_analysis_results": final_data,
        "heatmap_storage_path": heatmap_storage_path,
    }