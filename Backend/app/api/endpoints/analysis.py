# backend/app/api/endpoints/analysis.py (Correct and Complete for Task B2)
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import Dict, Annotated
import logging

# --- Import the AI service function ---
from ...services.image_analysis import analyze_chest_xray
# --- Import placeholders for future use (commented out) ---
# from ...services.report_generation import generate_report
# from ...services.patient_service import get_patient_history

# Get the logger instance
logger = logging.getLogger(__name__)

# --- Create the APIRouter instance ---
# This is crucial for main.py to find and include the routes
router = APIRouter()
# ------------------------------------

# Define the POST endpoint for the '/analyze/' path relative to the router prefix
@router.post("/")
async def run_analysis(
    # Parameters without default values first
    analysis_type: Annotated[str, Form(...)],
    patient_id: Annotated[int, Form(...)],
    image_file: Annotated[UploadFile, File(...)],
    # Parameters with default values last
    symptoms: Annotated[str, Form()] = ""
) -> Dict:
    """
    Receives image file, analysis type, patient ID, and symptoms.
    Calls the appropriate AI model (currently CheXNet for chest_xray).
    Returns the structured analysis results. Corresponds to Task B2.
    """
    logger.info(f"Received analysis request for patient {patient_id}. Type: {analysis_type}")

    # --- Read image file ---
    image_contents = await image_file.read()
    if not image_contents:
        logger.error("Error: Image file is empty.")
        raise HTTPException(status_code=400, detail="Image file is empty.")
    logger.info(f"Image file '{image_file.filename}' read successfully ({len(image_contents)} bytes).")

    # Initialize variables for results
    analysis_results = {}
    llm_report = "Report generation is not yet implemented." # Placeholder

    # --- Route to the correct analysis based on type ---
    if analysis_type == "chest_xray":
        try:
            logger.info("Calling chest x-ray analysis service (analyze_chest_xray)...")
            # --- Call the AI function from the service module ---
            image_analysis_output = analyze_chest_xray(image_bytes=image_contents)
            # Extract the results dictionary safely using .get()
            analysis_results = image_analysis_output.get("analysis_results", {})
            # --------------------------------------------------
            logger.info(f"Image analysis completed successfully.")

            # --- Placeholder for future LLM call ---
            # prompt_data = { ... }
            # llm_report = generate_report(prompt_data)
            # --------------------------------------

        except HTTPException as he:
            # If the AI service raised an HTTPException, pass it through
            logger.error(f"Error relayed from AI service: {he.detail}")
            raise he
        except Exception as e:
            # Catch any other unexpected errors during AI processing
            logger.exception(f"Unexpected error during AI analysis execution: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error during AI analysis: {e}")
    else:
        # Handle cases where the analysis type is not supported yet
        logger.warning(f"Analysis type '{analysis_type}' is not supported yet.")
        raise HTTPException(
            status_code=400,
            detail=f"Analysis type '{analysis_type}' is not supported yet."
        )

    # --- Placeholder for future database saving ---
    # try:
    #     logger.info(f"Saving analysis result for patient {patient_id}...")
    #     # await save_analysis_result(patient_id=patient_id, ...)
    # except Exception as e:
    #     logger.exception(f"Error saving result to database: {e}")
    # ---------------------------------------------

    # --- Construct the final response ---
    final_result = {
        "patient_id": patient_id,
        "analysis_type": analysis_type,
        "image_analysis_results": analysis_results, # Results from CheXNet
        "llm_report": llm_report # Placeholder text for now
    }
    logger.info("Sending final analysis result to the client.")
    return final_result