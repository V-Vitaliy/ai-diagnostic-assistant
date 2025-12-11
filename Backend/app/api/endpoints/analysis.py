from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import Dict, Annotated, Any, Optional
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
from app.db.models.chat_session import ChatSession # Added import

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
    session_id: Annotated[Optional[str], Form()] = None, # Added optional session_id
    symptoms: Annotated[str, Form()] = "",
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:

    logger.info(f"Start Analysis: Patient {patient_id}, Type {analysis_type}, Session {session_id}")

    # 1. Validate Patient
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Patient not found")

    # 2. Generate UUID for the analysis
    analysis_uuid = uuid.uuid4()

    # 3. Save Source File
    original_ext = image_file.filename.split(".")[-1] if "." in image_file.filename else "png"
    if analysis_type == "whole_body_ct" and not original_ext.endswith("gz"):
        pass
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

    # 5. Handle Heatmap
    final_data = image_analysis_output.get("analysis_results", image_analysis_output)
    heatmap_base64 = final_data.get("heatmap_base64", "")
    heatmap_storage_path = None

    if heatmap_base64:
        try:
            heatmap_bytes = base64.b64decode(heatmap_base64)
            heatmap_filename = f"{analysis_uuid}_heatmap.png"
            heatmap_full_path = f"{HEATMAP_DIR}/{heatmap_filename}"

            with open(heatmap_full_path, "wb") as f:
                f.write(heatmap_bytes)

            heatmap_storage_path = heatmap_full_path

            if "heatmap_base64" in final_data:
                del final_data["heatmap_base64"]

        except Exception as e:
            logger.error(f"Failed to save heatmap image: {e}")

    # 6. Save to Database
    new_analysis = AnalysisResult(
        id=analysis_uuid,
        patient_id=patient_id,
        analysis_type=analysis_type,
        symptoms_input=symptoms,
        image_storage_path=file_path,
        heatmap_storage_path=heatmap_storage_path,
        raw_model_outputs=final_data,
    )

    db.add(new_analysis)

    # --- 7. UPDATE CHAT HISTORY (Fix for persistence) ---
    if session_id:
        # Find the specific session
        res = await db.execute(select(ChatSession).where(ChatSession.session_id == session_id))
        chat_session = res.scalar_one_or_none()
    else:
        # Fallback: Find latest session for patient
        res = await db.execute(
            select(ChatSession)
            .where(ChatSession.patient_id == patient_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(1)
        )
        chat_session = res.scalar_one_or_none()

    if chat_session:
        current_history = list(chat_session.history_json) if chat_session.history_json else []

        # Add User Message (Upload context)
        user_msg = {
            "role": "user",
            "parts": [{"text": symptoms if symptoms else f"📎 Analiza pliku: {image_file.filename}"}],
            "file_label": f"Typ: {analysis_type}" # Custom field we will use in frontend
        }

        # Add System/AI Message (Result context)
        system_msg = {
            "role": "system_visualization",
            "parts": [{"text": "Wynik analizy"}],
            "analysis_id": str(analysis_uuid),
            "heatmap_storage_path": heatmap_storage_path,
            "image_analysis_results": final_data
        }

        current_history.append(user_msg)
        current_history.append(system_msg)

        # Update session
        chat_session.history_json = current_history
        db.add(chat_session) # Ensure it's marked for update

    await db.commit()
    await db.refresh(new_analysis)

    return {
        "id": new_analysis.id,
        "patient_id": patient_id,
        "analysis_type": analysis_type,
        "image_analysis_results": final_data,
        "heatmap_storage_path": heatmap_storage_path,
    }