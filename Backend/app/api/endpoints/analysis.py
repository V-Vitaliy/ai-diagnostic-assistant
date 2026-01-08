from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, BackgroundTasks
from typing import Dict, Annotated, Any, Optional
import logging
import shutil
import os
import uuid
import base64
import asyncio

# --- Database Imports ---
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db, AsyncSessionLocal # Make sure AsyncSessionLocal is exported in your session.py
from app.db.models.analysis_result import AnalysisResult
from app.db.models.patient import Patient
from app.db.models.chat_session import ChatSession

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

# --- BACKGROUND TASK ---
async def process_analysis_task(
    analysis_id: uuid.UUID,
    patient_id: int,
    analysis_type: str,
    file_path: str,
    file_content: bytes,
    original_filename: str,
    symptoms: str,
    session_id: Optional[str]
):
    """
    Background task to run AI analysis and update DB status.
    """
    logger.info(f"Background Task Started: Analysis {analysis_id}")

    # Create a NEW session for the background task
    async with AsyncSessionLocal() as db:
        try:
            # 1. Run AI Analysis
            image_analysis_output = {}
            try:
                if analysis_type == "chest_xray":
                    image_analysis_output = analyze_chest_xray(image_bytes=file_content)
                elif analysis_type == "extremity_xray":
                    image_analysis_output = analyze_extremity_xray(image_bytes=file_content)
                elif analysis_type == "ocr":
                    image_analysis_output = analyze_blood_image(image_bytes=file_content)
                elif analysis_type == "whole_body_ct":
                    # Note: analyze_whole_body_ct_3d is CPU/GPU intensive
                    image_analysis_output = analyze_whole_body_ct_3d(temp_file_path=file_path)
                else:
                    image_analysis_output = {"error": "Type not supported"}
            except Exception as e:
                logger.error(f"AI Service Failed inside task: {e}")
                image_analysis_output = {"error": str(e), "status": "failed"}

            # 2. Handle Heatmap
            final_data = image_analysis_output.get("analysis_results", image_analysis_output)
            heatmap_base64 = final_data.get("heatmap_base64", "")
            heatmap_storage_path = None

            if heatmap_base64:
                try:
                    heatmap_bytes = base64.b64decode(heatmap_base64)
                    heatmap_filename = f"{analysis_id}_heatmap.png"
                    heatmap_full_path = f"{HEATMAP_DIR}/{heatmap_filename}"

                    # Write file (sync operation in async func is okay for small files, or use aiofiles)
                    with open(heatmap_full_path, "wb") as f:
                        f.write(heatmap_bytes)

                    heatmap_storage_path = heatmap_full_path
                    if "heatmap_base64" in final_data:
                        del final_data["heatmap_base64"]
                except Exception as e:
                    logger.error(f"Failed to save heatmap image: {e}")

            # 3. Update AnalysisResult in DB
            result = await db.execute(select(AnalysisResult).where(AnalysisResult.id == analysis_id))
            db_analysis = result.scalar_one_or_none()

            if db_analysis:
                db_analysis.heatmap_storage_path = heatmap_storage_path
                db_analysis.raw_model_outputs = final_data

                # Check for errors in analysis output
                if "error" in image_analysis_output or final_data.get("status") == "failed":
                     db_analysis.status = "FAILED"
                else:
                     db_analysis.status = "COMPLETED"

                db.add(db_analysis)

            # 4. Update Chat Session (History)
            if session_id:
                res = await db.execute(select(ChatSession).where(ChatSession.session_id == session_id))
                chat_session = res.scalar_one_or_none()
            else:
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
                    "parts": [{"text": symptoms if symptoms else f"📎 Analiza pliku: {original_filename}"}],
                    "file_label": f"Typ: {analysis_type}"
                }

                # Add System/AI Message (Result context)
                system_msg = {
                    "role": "system_visualization",
                    "parts": [{"text": "Wynik analizy"}],
                    "analysis_id": str(analysis_id),
                    "heatmap_storage_path": heatmap_storage_path,
                    "image_analysis_results": final_data
                }

                current_history.append(user_msg)
                current_history.append(system_msg)

                chat_session.history_json = current_history
                db.add(chat_session)

            await db.commit()
            logger.info(f"Background Task Completed: Analysis {analysis_id}")

        except Exception as e:
            logger.error(f"CRITICAL BACKGROUND TASK ERROR: {e}")
            await db.rollback()
            # Try to set status to FAILED if possible
            try:
                 result = await db.execute(select(AnalysisResult).where(AnalysisResult.id == analysis_id))
                 fail_analysis = result.scalar_one_or_none()
                 if fail_analysis:
                     fail_analysis.status = "FAILED"
                     fail_analysis.raw_model_outputs = {"error": str(e)}
                     await db.commit()
            except:
                pass

# --- ENDPOINTS ---

@router.post("/")
async def run_analysis(
    background_tasks: BackgroundTasks,
    analysis_type: Annotated[str, Form(...)],
    patient_id: Annotated[int, Form(...)],
    image_file: Annotated[UploadFile, File(...)],
    session_id: Annotated[Optional[str], Form()] = None,
    symptoms: Annotated[str, Form()] = "",
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:

    logger.info(f"Start Analysis Request: Patient {patient_id}, Type {analysis_type}")

    # 1. Validate Patient
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Patient not found")

    # 2. Generate UUID
    analysis_uuid = uuid.uuid4()

    # 3. Save Source File Immediately
    original_ext = image_file.filename.split(".")[-1] if "." in image_file.filename else "png"
    source_filename = f"{analysis_uuid}.{original_ext}"
    file_path = f"{UPLOAD_DIR}/{source_filename}"

    file_content = await image_file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Empty file")

    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    # 4. Create DB Entry with PENDING status
    new_analysis = AnalysisResult(
        id=analysis_uuid,
        patient_id=patient_id,
        analysis_type=analysis_type,
        symptoms_input=symptoms,
        image_storage_path=file_path,
        status="PENDING", # Requires DB Migration
        raw_model_outputs={},
    )

    db.add(new_analysis)
    await db.commit()
    await db.refresh(new_analysis)

    # 5. Delegate processing to Background Task
    background_tasks.add_task(
        process_analysis_task,
        analysis_uuid,
        patient_id,
        analysis_type,
        file_path,
        file_content,
        image_file.filename,
        symptoms,
        session_id
    )

    # 6. Return ID immediately
    return {
        "id": new_analysis.id,
        "status": "PENDING",
        "message": "Analysis started in background"
    }

@router.get("/{analysis_id}")
async def get_analysis_status(
    analysis_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:

    # Try to parse UUID
    try:
        uuid_obj = uuid.UUID(analysis_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    result = await db.execute(select(AnalysisResult).where(AnalysisResult.id == uuid_obj))
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    response = {
        "id": analysis.id,
        "status": analysis.status, # PENDING, COMPLETED, FAILED
        "patient_id": analysis.patient_id,
        "analysis_type": analysis.analysis_type,
    }

    # Only include heavy results if completed
    if analysis.status == "COMPLETED":
        response["image_analysis_results"] = analysis.raw_model_outputs
        response["heatmap_storage_path"] = analysis.heatmap_storage_path
    elif analysis.status == "FAILED":
        response["error"] = analysis.raw_model_outputs.get("error", "Unknown error")

    return response