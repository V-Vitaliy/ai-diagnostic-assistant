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

# --- Import from 'pytorch-gradcam' ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fracture_model_path = "/app/app/services/models/model.pt"
device = torch.device("cpu")

# --- CheXNet Model & Transforms ---
chest_model = None
chest_transform = None
chest_model_target_layer = None #  Added target layer

def load_chest_model():
    """Loads the CheXNet model, transforms, and target layer on the first call."""
    global chest_model, chest_transform, chest_model_target_layer # <-- UPDATED
    if chest_model is None:
        try:
            logger.info("Loading pre-trained CheXNet model (densenet121-res224-all)...")
            chest_model = xrv.models.DenseNet(weights="densenet121-res224-all")
            chest_model.eval()

            # ---Define the target layer for Grad-CAM ---
            # We target the last layer of the 'features' block
            chest_model_target_layer = [chest_model.features[-1]]

            logger.info("CheXNet model and target layer loaded successfully.")

            # Transforms for CheXNet (1-channel input, normalized to [0, 1])
            chest_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(), # Scales image pixels to the range [0.0, 1.0]
            ])
            logger.info("Transforms for CheXNet configured.")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load CheXNet model: {e}")
            raise RuntimeError(f"Failed to load AI model (CheXNet): {e}")

# --- Fracture Model & Transforms ---
fracture_model = None
fracture_model_target_layer = None

def load_fracture_model():
    """Loads the custom Fracture Detection model."""
    # (This function is unchanged)
    global fracture_model, fracture_model_target_layer
    if fracture_model is None:
        try:
            logger.info("Loading pre-trained Fracture model (Custom Densenet)...")
            fracture_model = PretrainedDensenet()
            fracture_model.load_state_dict(torch.load(fracture_model_path, map_location=device))
            fracture_model.eval()
            fracture_model_target_layer = [fracture_model.features[-1]]
            logger.info("Fracture model loaded successfully.")
        except FileNotFoundError:
             logger.error(f"CRITICAL ERROR: Model file not found at {fracture_model_path}.")
             raise RuntimeError(f"Model file not found: {fracture_model_path}")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load Fracture model: {e}")
            raise RuntimeError(f"Failed to load AI model (Fracture): {e}")

# --- Utility Function  ---
def convert_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a Base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# ---
# --- ***  CHEST X-RAY ANALYSIS FUNCTION *** ---
# ---
def analyze_chest_xray(image_bytes: bytes) -> dict:
    """
    Analyzes chest x-ray for pathologies AND generates a Grad-CAM heatmap
    for the pathology with the highest probability.
    """
    # 1. Load model if not already loaded
    if chest_model is None or chest_transform is None:
        load_chest_model()
        if chest_model is None or chest_transform is None:
             raise HTTPException(status_code=503, detail="CheXNet model is temporarily unavailable.")

    try:
        # 2. Preprocess the image
        logger.info("Starting image preprocessing for CheXNet (1 channel)...")
        image = Image.open(io.BytesIO(image_bytes)).convert('L') # Grayscale
        img_tensor_norm = chest_transform(image) # Normalized tensor [0, 1]

        # We need the 3-channel version for visualization later
        vis_rgb = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_GRAY2RGB)
        vis_rgb = np.float32(vis_rgb) / 255

        # Add batch dimension
        img_tensor_norm = img_tensor_norm.unsqueeze(0) # Add batch: [1, 1, 224, 224]

        # ---    Enable gradients for Grad-CAM ---
        img_tensor_norm.requires_grad_(True)

        # 3. Get Model Predictions
        outputs = chest_model(img_tensor_norm)
        probabilities = torch.sigmoid(outputs).cpu() # Keep as tensor

        # --- Find the pathology with the highest probability ---
        highest_prob = torch.max(probabilities)
        highest_prob_index = torch.argmax(probabilities)
        highest_pathology_name = chest_model.pathologies[highest_prob_index]

        logger.info(f"CheXNet prediction obtained. Highest prob: {highest_pathology_name} ({highest_prob:.3f})")

        # --- Generate Grad-CAM for the highest probability pathology ---
        logger.info(f"Generating Grad-CAM for '{highest_pathology_name}'...")

        cam = GradCAM(model=chest_model, target_layers=chest_model_target_layer)

        # Target the class (pathology) with the highest score
        targets = [ClassifierOutputTarget(highest_prob_index.item())]

        grayscale_cam = cam(input_tensor=img_tensor_norm, targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Get the first heatmap

        # --- Create overlay ---
        # We use the resized 3-channel image (vis_rgb) we prepared earlier
        cam_image_overlay = show_cam_on_image(vis_rgb, grayscale_cam, use_rgb=True, image_weight=0.6)
        heatmap_image = Image.fromarray(cam_image_overlay)
        heatmap_base64 = convert_to_base64(heatmap_image)
        logger.info("CheXNet heatmap successfully generated and encoded.")

        # --- 4. Format results ---
        results = {}
        threshold = 0.1 # Probability threshold
        probabilities_np = probabilities.detach().numpy()[0] # Convert to numpy for iteration

        for i, pathology in enumerate(chest_model.pathologies):
            prob = float(probabilities_np[i])
            if prob >= threshold:
                results[pathology] = round(prob, 3)

        if not results:
             results = {"status": f"No findings with probability >= {threshold}."}

        # ---  Add the new heatmap fields to the results ---
        results["heatmap_base64"] = heatmap_base64
        results["heatmap_target"] = highest_pathology_name # So frontend knows what heatmap shows

        logger.info(f"CheXNet analysis complete. Found {len(results)-2} pathologies.") # -2 for heatmap fields
        return {"analysis_results": results}

    except Exception as e:
        logger.exception(f"Error during CheXNet image analysis or CAM generation: {e}")
        raise HTTPException(status_code=500, detail=f"AI model processing error (CheXNet): {e}")


# ---
# --- Fracture Analysis Function (Unchanged) ---
# ---
def analyze_extremity_xray(image_bytes: bytes) -> dict:
    """
    Analyzes extremity x-ray for fractures and generates a Grad-CAM heatmap
    using the 'pytorch-gradcam' library.
    """
    # (This function is unchanged from our previous step)
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

        inp_norm.requires_grad_(True)
        output_logits = fracture_model(inp_norm)
        prob = torch.sigmoid(output_logits).item()
        logger.info(f"Fracture model prediction obtained. Probability: {prob:.3f}")

        logger.info("Generating Grad-CAM heatmap with 'pytorch-gradcam'...")
        cam = GradCAM(model=fracture_model, target_layers=fracture_model_target_layer)
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=inp_norm, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        vis_input_image = inp.squeeze(0).permute(1, 2, 0).numpy()
        vis_rgb = (vis_input_image[:, :, 0] * 255).astype(np.uint8)
        vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_GRAY2RGB)
        vis_rgb = np.float32(vis_rgb) / 255

        cam_image_overlay = show_cam_on_image(vis_rgb, grayscale_cam, use_rgb=True, image_weight=0.6)
        heatmap_image = Image.fromarray(cam_image_overlay)
        heatmap_base64 = convert_to_base64(heatmap_image)
        logger.info("Heatmap successfully generated and encoded to Base64.")

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


