import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image
import io
import logging
import torchvision.transforms as transforms
from fastapi import HTTPException # Import for handling API errors

# Setup logging to see messages in Docker console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model for Chest X-Ray Analysis ---

# Global variable to hold the model (lazy loading)
model = None
transform = None # Add global variable for transforms

def load_model():
    """Loads the model and transforms on the first call."""
    global model, transform
    if model is None:
        try:
            logger.info("Loading pre-trained CheXNet model (densenet121-res224-all)...")
            # Using 'all' weights for the maximum number of pathologies.
            # Note: Weights will be downloaded on the first container run!
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            model.eval() # Set the model to evaluation mode (disables dropout, etc.)
            logger.info("CheXNet model loaded successfully and ready for inference.")

            # Define the standard transformations expected by the model
            # 224x224 is the standard input size for this model
            transform = transforms.Compose([
                # Resize to 256x256 (common practice before CenterCrop)
                transforms.Resize(256),
                # Crop the center 224x224 portion
                transforms.CenterCrop(224),
                # Convert PIL image to PyTorch tensor (pixel values scaled to [0, 1])
                transforms.ToTensor(),
                # Normalize the image (mean and std dev from ImageNet)
                # This model (densenet121-res224-all) expects 3 channels
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info("Transforms for the model configured.")

        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load CheXNet model/setup transforms: {e}")
            # Keep model = None in case of error
            raise RuntimeError(f"Failed to load AI model: {e}") # Propagate the error


def analyze_chest_xray(image_bytes: bytes) -> dict:
    """
    Takes chest x-ray image bytes, analyzes using CheXNet,
    and returns a dictionary with pathology probabilities.
    """
    # Lazy load the model and transforms on the first function call
    if model is None or transform is None:
        load_model()
        # Re-check in case loading failed
        if model is None or transform is None:
             raise HTTPException(status_code=503, detail="AI model is temporarily unavailable (failed to load).")

    try:
        logger.info("Starting image preprocessing for CheXNet...")
        # 1. Open image from bytes using Pillow
        # .convert('RGB') - Important! The model expects 3 channels
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 2. Apply the transformations defined above
        img_tensor = transform(image)
        logger.debug(f"Tensor shape after transforms: {img_tensor.shape}") # Expect [3, 224, 224]

        # 3. Add batch dimension
        # The model expects a batch of images, even if it's just one
        # [3, 224, 224] -> [1, 3, 224, 224]
        img_tensor = img_tensor.unsqueeze(0)
        logger.debug(f"Tensor shape with batch dimension: {img_tensor.shape}")

        # 4. Run the model to get raw predictions (logits)
        # torch.no_grad() disables gradient calculation, speeds up inference and saves memory
        with torch.no_grad():
            outputs = model(img_tensor)
            # Apply sigmoid because a patient can have MULTIPLE pathologies simultaneously
            # (multi-label classification). Sigmoid converts logits to probabilities [0, 1] for each class.
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0] # [0] because batch size is 1
        logger.info("CheXNet model prediction obtained.")

        # 5. Format the result into a user-friendly dictionary
        results = {}
        threshold = 0.1 # Probability threshold (10%) - pathologies below this won't be reported
        logger.debug(f"Model pathologies list: {model.pathologies}")

        # Iterate through all pathologies the model knows
        for i, pathology in enumerate(model.pathologies):
            prob = float(probabilities[i])
            # If probability is above threshold, add to results
            if prob >= threshold:
                results[pathology] = round(prob, 3) # Round to 3 decimal places
                logger.debug(f"Detected: {pathology} (Probability: {prob:.3f})")

        if not results:
             logger.info(f"No pathologies found with probability >= {threshold}.")
             # Can return a specific status or leave empty
             return {"analysis_results": {"status": f"No findings with probability >= {threshold}."}}

        logger.info(f"Analysis complete. Found {len(results)} pathologies above threshold {threshold}.")
        return {"analysis_results": results}

    except Exception as e:
        # Log the full traceback for debugging
        logger.exception(f"Error during CheXNet image analysis: {e}")
        # Return a standardized API error
        # Use HTTPException so FastAPI returns the correct status code
        raise HTTPException(status_code=500, detail=f"AI model processing error: {e}")

