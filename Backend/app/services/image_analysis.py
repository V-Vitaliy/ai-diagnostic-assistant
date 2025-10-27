import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image
import io
import logging
import torchvision.transforms as transforms
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
transform = None

def load_model():
    """Loads the model and transforms on the first call."""
    global model, transform
    if model is None:
        try:
            logger.info("Loading pre-trained CheXNet model (densenet121-res224-all)...")
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            model.eval()
            logger.info("CheXNet model loaded successfully.")

            # --- UPDATED TRANSFORMS FOR 1 CHANNEL ---
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # 1. Ensure image is Grayscale BEFORE converting to tensor
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # 2. Use Normalize suitable for 1 channel (e.g., ImageNet mean/std for grayscale)
                #    Often, medical images use simple normalization [0, 1] or specific windowing.
                #    Let's try normalizing with the mean of ImageNet means/stds as a starting point.
                #    mean = 0.485*0.299 + 0.456*0.587 + 0.406*0.114 approx 0.449
                #    std = sqrt( (0.229*0.299)**2 + (0.224*0.587)**2 + (0.225*0.114)**2 ) approx 0.157 ???
                #    OR just use the values torchxrayvision might implicitly use or simpler [0.5], [0.5]
                #    Let's use [0.5], [0.5] for normalization to [-1, 1] range after ToTensor brings to [0, 1]
                # transforms.Normalize(mean=[0.5], std=[0.5])
                # Alternatively, skip normalization if ToTensor() brings to [0, 1] which might be sufficient
                # If skipping Normalize, comment out the line above.
            ])
            logger.info("Transforms configured (Resize, Crop, Grayscale, ToTensor to [0,1]).")

        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load CheXNet model/setup transforms: {e}")
            raise RuntimeError(f"Failed to load AI model: {e}")


def analyze_chest_xray(image_bytes: bytes) -> dict:
    """
    Takes chest x-ray image bytes (expects grayscale), analyzes using CheXNet,
    and returns a dictionary with pathology probabilities.
    """
    if model is None or transform is None:
        load_model()
        if model is None or transform is None:
             raise HTTPException(status_code=503, detail="AI model is temporarily unavailable (failed to load).")

    try:
        logger.info("Starting image preprocessing for CheXNet (1 channel)...")
        # 1. Open image from bytes using Pillow
        # --- CHANGED: Convert to 'L' (Grayscale) instead of 'RGB' ---
        image = Image.open(io.BytesIO(image_bytes)).convert('L')

        # 2. Apply the updated transformations
        img_tensor = transform(image)
        logger.debug(f"Tensor shape after transforms: {img_tensor.shape}") # Expect [1, 224, 224]

        # 3. Add batch dimension
        # [1, 224, 224] -> [1, 1, 224, 224]
        img_tensor = img_tensor.unsqueeze(0)
        logger.debug(f"Tensor shape with batch dimension: {img_tensor.shape}")

        # --- REMOVED: No need to repeat channels ---
        # if img_tensor.shape[1] == 1:
        #      img_tensor = img_tensor.repeat(1, 3, 1, 1) # REMOVED

        # 4. Run the model
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        logger.info("CheXNet model prediction obtained.")

        # 5. Format the result
        results = {}
        threshold = 0.1
        logger.debug(f"Model pathologies list: {model.pathologies}")
        for i, pathology in enumerate(model.pathologies):
            prob = float(probabilities[i])
            if prob >= threshold:
                results[pathology] = round(prob, 3)
                logger.debug(f"Detected: {pathology} (Probability: {prob:.3f})")

        if not results:
             logger.info(f"No pathologies found with probability >= {threshold}.")
             return {"analysis_results": {"status": f"No findings with probability >= {threshold}."}}

        logger.info(f"Analysis complete. Found {len(results)} pathologies above threshold {threshold}.")
        return {"analysis_results": results}

    except Exception as e:
        logger.exception(f"Error during CheXNet image analysis: {e}")
        raise HTTPException(status_code=500, detail=f"AI model processing error: {e}")