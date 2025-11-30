# backend/app/api/endpoints/analysis.py
# (Adds routing for brain_ct and handles temp file saving)

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import Dict, Annotated
import logging
import tempfile # For creating temporary files
import os # For deleting temporary files

# --- Import ALL AI service functions ---
from ...services.image_analysis import (
    analyze_chest_xray, 
    analyze_extremity_xray,
    analyze_whole_body_ct_3d

)
from ...services.ocr_service import analyze_blood_image

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/")
async def run_analysis(
    analysis_type: Annotated[str, Form(...)],
    patient_id: Annotated[int, Form(...)],
    image_file: Annotated[UploadFile, File(...)],
    symptoms: Annotated[str, Form()] = ""
) -> Dict:
    """
    Receives data, routes to the appropriate AI model based on analysis_type,
    and returns the analysis result.
    """
    logger.info(f"Received analysis request for patient {patient_id}. Type: {analysis_type}")

    image_analysis_output = {} 
    llm_report = "Report generation is not yet implemented." # Placeholder
    
    # --- Logic for 2D PNG/JPG files  ---
    if analysis_type in ["chest_xray", "extremity_xray","ocr"]:
        image_contents = await image_file.read()
        if not image_contents:
            logger.error("Error: Image file is empty.")
            raise HTTPException(status_code=400, detail="Image file is empty.")
        
        try:
            if analysis_type == "chest_xray":
                logger.info("Routing to chest x-ray analysis service...")
                image_analysis_output = analyze_chest_xray(image_bytes=image_contents)
            
            elif analysis_type == "extremity_xray":
                logger.info("Routing to extremity (fracture) analysis service...")
                image_analysis_output = analyze_extremity_xray(image_bytes=image_contents)

            elif analysis_type == "ocr":
                logger.info("Routing to OCR blood analysis service...")
                image_analysis_output = analyze_blood_image(image_bytes=image_contents)

            logger.info(f"2D Image analysis completed successfully for type: {analysis_type}.")

        except HTTPException as he:
             logger.error(f"Error relayed from AI service: {he.detail}")
             raise he
        except Exception as e:
            logger.exception(f"Unexpected error during 2D analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error during 2D analysis: {e}")

    # --- NEW Logic for 3D .nii.gz files ---
    elif analysis_type == "whole_body_ct":
        if not image_file.filename.endswith(('.nii', '.nii.gz')):
            raise HTTPException(status_code=400, detail="Invalid file type for brain_ct. Expected .nii or .nii.gz")

        # Save 3D file to a temporary location
        temp_file_path = None
        try:
            # Create a named temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=image_file.filename) as tmp:
                await tmp.write(await image_file.read())
                temp_file_path = tmp.name # Get the path
            
            logger.info(f"3D file saved to temporary path: {temp_file_path}")

            # Call the AI service with the FILE PATH
            image_analysis_output = analyze_whole_body_ct_3d(temp_file_path=temp_file_path)
            
            logger.info(f"3D Image analysis (MONAI) completed successfully.")
        
        except HTTPException as he:
             logger.error(f"Error relayed from 3D AI service: {he.detail}")
             raise he
        except Exception as e:
            logger.exception(f"Unexpected error during 3D analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error during 3D analysis: {e}")
        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.info(f"Temporary file {temp_file_path} deleted.")

    else:
        # Handle unsupported analysis types
        logger.warning(f"Analysis type '{analysis_type}' is not supported yet.")
        raise HTTPException(
            status_code=400,
            detail=f"Analysis type '{analysis_type}' is not supported yet."
        )
    # --- End of Router Logic ---

    # Construct the final response
    final_result = {
        "patient_id": patient_id,
        "analysis_type": analysis_type,
        "image_analysis_results": image_analysis_output.get("analysis_results", {}),
        "llm_report": llm_report
    }
    logger.info("Sending final analysis result to the client.")
    return final_result