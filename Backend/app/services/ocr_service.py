import easyocr
import numpy as np
from PIL import Image
import io
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize reader lazily to save memory/startup time
reader = None

def get_reader():
    global reader
    if reader is None:
        logger.info("Initializing EasyOCR reader...")
        # Load English and Polish models
        # gpu=False ensures compatibility if CUDA is not available
        reader = easyocr.Reader(['pl', 'en'], gpu=False)
        logger.info("EasyOCR reader initialized.")
    return reader

def parse_blood_test_results(text_list: list) -> dict:
    """
    Parses a list of strings from OCR to find blood test parameters.
    Returns a dictionary {parameter: value}.
    """
    results = {}

    # Define patterns for common blood parameters (Polish/English names)
    # Regex logic:
    # 1. Match parameter name (e.g., 'HGB' or 'Hemoglobina')
    # 2. .*? - Skip any characters in between (dots, spaces, units)
    # 3. (\d+[.,]\d+) - Capture the number (can have dot or comma)
    patterns = {
        "WBC (Leukocyty)": r"(WBC|Leukocyty|White Blood Cells).*?(\d+[.,]\d+)",
        "RBC (Erytrocyty)": r"(RBC|Erytrocyty|Red Blood Cells).*?(\d+[.,]\d+)",
        "HGB (Hemoglobina)": r"(HGB|Hemoglobina|Hemoglobin).*?(\d+[.,]\d+)",
        "HCT (Hematokryt)": r"(HCT|Hematokryt|Hematocrit).*?(\d+[.,]\d+)",
        "MCV": r"(MCV).*?(\d+[.,]\d+)",
        "MCH": r"(MCH).*?(\d+[.,]\d+)",
        "MCHC": r"(MCHC).*?(\d+[.,]\d+)",
        "PLT (Płytki krwi)": r"(PLT|Płytki krwi|Platelets).*?(\d+)",
    }

    # Combine all text lines into one string to handle cases where
    # value might be on the same line or slightly offset.
    # However, searching line-by-line is often safer for table structures.
    # Let's try searching in the full text first.
    full_text = " ".join(text_list)

    for param, pattern in patterns.items():
        # Search for the pattern (case insensitive)
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            value_str = match.group(2)
            # Normalize number format: replace comma with dot for float conversion
            value_str = value_str.replace(',', '.')
            try:
                value = float(value_str)
                results[param] = value
            except ValueError:
                continue

    return results

def analyze_blood_image(image_bytes: bytes) -> dict:
    """
    Main function: Takes image bytes, runs OCR, and extracts blood test parameters.
    """
    try:
        reader = get_reader()

        # Convert bytes to image format compatible with EasyOCR (numpy array)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        logger.info("Running OCR on image...")
        # detail=0 returns just the list of recognized text strings
        # paragraph=False (default) treats each line separately
        raw_text_list = reader.readtext(image_np, detail=0)

        logger.info(f"OCR complete. Found {len(raw_text_list)} text elements.")

        # Parse the text to find medical data
        parsed_data = parse_blood_test_results(raw_text_list)

        if not parsed_data:
            logger.warning("No blood parameters identified in the text.")
            return {
                "status": "No recognizable blood parameters found.",
                "raw_text_sample": raw_text_list[:5] # Return sample for debugging
            }

        logger.info(f"Successfully parsed {len(parsed_data)} parameters.")
        return {
            "blood_analysis_results": parsed_data,
            # We can also return the raw text if needed for LLM later
            # "full_text": raw_text_list
        }

    except Exception as e:
        logger.exception(f"Error during OCR analysis: {e}")
        # In a real app service, we might raise an exception to be caught by API
        raise e