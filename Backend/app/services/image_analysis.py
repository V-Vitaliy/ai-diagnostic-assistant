# backend/app/services/image_analysis.py (Updated)

import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image
import io
import logging
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Import for functional transforms
from fastapi import HTTPException

# --- Import custom model ---
# Use relative import from the new 'models' sub-directory
from .models.CustomModel import PretrainedDensenet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CheXNet Model (for chest x-ray) ---
chest_model = None
chest_transform = None
# Path inside the Docker container
fracture_model_path = "/app/app/services/models/model.pt"

def load_chest_model():
    """Loads the CheXNet model and transforms on the first call."""
    global chest_model, chest_transform
    if chest_model is None:
        try:
            logger.info("Loading pre-trained CheXNet model (densenet121-res224-all)...")
            chest_model = xrv.models.DenseNet(weights="densenet121-res224-all")
            chest_model.eval()
            logger.info("CheXNet model loaded successfully.")

            chest_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
            logger.info("Transforms for CheXNet configured.")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load CheXNet model: {e}")
            raise RuntimeError(f"Failed to load AI model (CheXNet): {e}")

# --- Fracture Model (for extremities) ---
fracture_model = None
device = torch.device("cpu") # Run on CPU

def load_fracture_model():
    """Loads the custom Fracture Detection model on the first call."""
    global fracture_model
    if fracture_model is None:
        try:
            logger.info("Loading pre-trained Fracture model (Custom Densenet)...")
            fracture_model = PretrainedDensenet()
            fracture_model.load_state_dict(torch.load(fracture_model_path, map_location=device))
            fracture_model.eval()
            logger.info("Fracture model loaded successfully.")
        except FileNotFoundError:
             logger.error(f"CRITICAL ERROR: Model file not found at {fracture_model_path}. Did you update the Dockerfile?")
             raise RuntimeError(f"Model file not found: {fracture_model_path}")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load Fracture model: {e}")
            raise RuntimeError(f"Failed to load AI model (Fracture): {e}")


# --- Analysis function for CheXNet ---
def analyze_chest_xray(image_bytes: bytes) -> dict:
    """Analyzes chest x-ray for pathologies."""
    if chest_model is None or chest_transform is None:
        load_chest_model()
        if chest_model is None or chest_transform is None:
             raise HTTPException(status_code=503, detail="CheXNet model is temporarily unavailable.")

    try:
        logger.info("Starting image preprocessing for CheXNet (1 channel)...")
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        img_tensor = chest_transform(image)
        min_val, max_val = torch.min(img_tensor), torch.max(img_tensor)
        logger.debug(f"Tensor value range after ToTensor: [{min_val:.2f}, {max_val:.2f}]")
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = chest_model(img_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        logger.info("CheXNet model prediction obtained.")

        results = {}
        threshold = 0.1
        for i, pathology in enumerate(chest_model.pathologies):
            prob = float(probabilities[i])
            if prob >= threshold:
                results[pathology] = round(prob, 3)
        
        if not results:
             return {"analysis_results": {"status": f"No findings with probability >= {threshold}."}}

        logger.info(f"CheXNet analysis complete. Found {len(results)} pathologies.")
        return {"analysis_results": results}

    except Exception as e:
        logger.exception(f"Error during CheXNet image analysis: {e}")
        raise HTTPException(status_code=500, detail=f"AI model processing error (CheXNet): {e}")


# --- NEW Analysis function for Fracture Detection ---
def analyze_extremity_xray(image_bytes: bytes) -> dict:
    """
    Analyzes extremity x-ray for fractures using the custom model.
    Replicates the exact preprocessing from the test script.
    """
    if fracture_model is None:
        load_fracture_model()
        if fracture_model is None:
            raise HTTPException(status_code=503, detail="Fracture model is temporarily unavailable.")
    
    try:
        logger.info("Starting image preprocessing for Fracture model (2 channels)...")
        # 1. Open image and convert to 'LA' (2 channels)
        image = Image.open(io.BytesIO(image_bytes)).convert('LA')

        # 2. Replicate transformations from test script
        tensor_img = TF.to_tensor(image) # Shape [2, H, W]
        tensor_img = TF.resize(tensor_img, [224, 224]) # Shape [2, 224, 224]
        inp = tensor_img.unsqueeze(0) # Shape [1, 2, 224, 224]

        # 3. Replicate manual normalization
        inp = (inp - 0.456) / 0.224
        logger.debug(f"Fracture model input tensor shape: {inp.shape}")

        # 4. Get prediction
        with torch.no_grad():
            output = fracture_model(inp)
            prob = torch.sigmoid(output).item() # .item() as it's binary classification
        
        logger.info(f"Fracture model prediction obtained. Probability: {prob:.3f}")

        # 5. Format result
        threshold = 0.5 # Standard threshold for binary classification
        finding = "Fracture detected" if prob > threshold else "No fracture detected"

        return {
            "analysis_results": {
                "finding": finding,
                "fracture_probability": round(prob, 3)
            }
        }
    
    except Exception as e:
        logger.exception(f"Error during Fracture model image analysis: {e}")
        raise HTTPException(status_code=500, detail=f"AI model processing error (Fracture): {e}")