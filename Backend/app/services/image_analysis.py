# backend/app/services/image_analysis.py (Updated to use pytorch-gradcam)

import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image
import io
import logging
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from fastapi import HTTPException
import cv2
import base64

# --- Import custom model ---
from .models.CustomModel import PretrainedDensenet

# --- Import from the NEW library 'pytorch-gradcam' ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# We need a target wrapper for classification models
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CheXNet Model (for chest x-ray) ---
chest_model = None
chest_transform = None
fracture_model_path = "/app/app/services/models/model.pt"
device = torch.device("cpu")

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
fracture_model_target_layer = None

def load_fracture_model():
    """Loads the custom Fracture Detection model and defines the target layer for CAM."""
    global fracture_model, fracture_model_target_layer
    if fracture_model is None:
        try:
            logger.info("Loading pre-trained Fracture model (Custom Densenet)...")
            fracture_model = PretrainedDensenet()
            fracture_model.load_state_dict(torch.load(fracture_model_path, map_location=device))
            fracture_model.eval()
            # This is the target layer for Grad-CAM
            fracture_model_target_layer = [fracture_model.features[-1]]
            logger.info("Fracture model loaded successfully.")
        except FileNotFoundError:
             logger.error(f"CRITICAL ERROR: Model file not found at {fracture_model_path}.")
             raise RuntimeError(f"Model file not found: {fracture_model_path}")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load Fracture model: {e}")
            raise RuntimeError(f"Failed to load AI model (Fracture): {e}")

def convert_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a Base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# --- Analysis function for CheXNet ---
def analyze_chest_xray(image_bytes: bytes) -> dict:
    """Analyzes chest x-ray for pathologies."""
    if chest_model is None or chest_transform is None:
        load_chest_model()
        if chest_model is None or chest_transform is None:
             raise HTTPException(status_code=503, detail="CheXNet model is temporarily unavailable.")

    # (Rest of the function logic is unchanged)
    try:
        logger.info("Starting image preprocessing for CheXNet (1 channel)...")
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        img_tensor = chest_transform(image)
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
             logger.info(f"No pathologies found with probability >= {threshold}.")
             return {"analysis_results": {"status": f"No findings with probability >= {threshold}."}}

        logger.info(f"CheXNet analysis complete. Found {len(results)} pathologies.")
        return {"analysis_results": results}

    except Exception as e:
        logger.exception(f"Error during CheXNet image analysis: {e}")
        raise HTTPException(status_code=500, detail=f"AI model processing error (CheXNet): {e}")


# --- UPDATED Analysis function for Fracture Detection ---
def analyze_extremity_xray(image_bytes: bytes) -> dict:
    """
    Analyzes extremity x-ray for fractures and generates a Grad-CAM heatmap
    using the 'pytorch-gradcam' library.
    """
    if fracture_model is None:
        load_fracture_model()
        if fracture_model is None:
            raise HTTPException(status_code=503, detail="Fracture model is temporarily unavailable.")

    try:
        logger.info("Starting image preprocessing for Fracture model (2 channels)...")
        image = Image.open(io.BytesIO(image_bytes)).convert('LA') # 2-channel
        tensor_img = TF.to_tensor(image) # Shape [2, H, W], range [0, 1]
        tensor_img_resized = TF.resize(tensor_img, [224, 224])
        inp = tensor_img_resized.unsqueeze(0) # Shape [1, 2, 224, 224]
        inp_norm = (inp - 0.456) / 0.224 # Normalized input for the model
        logger.debug(f"Fracture model input tensor shape: {inp_norm.shape}")

        # --- Get probability (we need to do this manually) ---
        # Enable gradients for the input tensor, required by GradCAM
        inp_norm.requires_grad_(True)

        output_logits = fracture_model(inp_norm)
        # Apply sigmoid to the raw logits
        prob = torch.sigmoid(output_logits).item()
        logger.info(f"Fracture model prediction obtained. Probability: {prob:.3f}")

        # --- Generate Grad-CAM Heatmap (using pytorch-gradcam) ---
        logger.info("Generating Grad-CAM heatmap with 'pytorch-gradcam'...")

        cam = GradCAM(model=fracture_model, target_layers=fracture_model_target_layer)

        # Define the target: For a binary classifier with one output,
        # we target the 0-th index (the only output score).
        targets = [ClassifierOutputTarget(0)]

        # You can also pass aug_smooth=True and eigen_smooth=True for better quality
        grayscale_cam = cam(input_tensor=inp_norm, targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Get the first (and only) heatmap

        # --- Create overlay ---
        # Convert 2-channel [0,1] tensor back to a 3-channel RGB image for overlay
        vis_input_image = inp.squeeze(0).permute(1, 2, 0).numpy() # Shape [224, 224, 2]
        # We use the L-channel (luminance) for all 3 RGB channels
        vis_rgb = (vis_input_image[:, :, 0] * 255).astype(np.uint8)
        vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_GRAY2RGB)
        vis_rgb = np.float32(vis_rgb) / 255 # Normalize back to [0, 1]
        
        cam_image_overlay = show_cam_on_image(vis_rgb, grayscale_cam, use_rgb=True, image_weight=0.6)
        heatmap_image = Image.fromarray(cam_image_overlay)
        
        # --- Encode heatmap to Base64 ---
        heatmap_base64 = convert_to_base64(heatmap_image)
        logger.info("Heatmap successfully generated and encoded to Base64.")

        # --- Format result ---
        threshold = 0.5
        finding = "Fracture detected" if prob > threshold else "No fracture detected"

        return {
            "analysis_results": {
                "finding": finding,
                "fracture_probability": round(prob, 3),
                "heatmap_base64": heatmap_base64
            }
        }
    
    except Exception as e:
        logger.exception(f"Error during Fracture model image analysis or CAM generation: {e}")
        raise HTTPException(status_code=500, detail=f"AI model processing error (Fracture/CAM): {e}")