import os
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
from pathlib import Path


# --- Import custom model ---
from .models.CustomModel import PretrainedDensenet

# --- Import from 'pytorch-gradcam' ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- NEW: Import MONAI, Nibabel, ImageIO ---
import monai
from monai.bundle import ConfigParser, ConfigWorkflow
from monai.transforms import LoadImage, Compose, EnsureChannelFirst, ScaleIntensity, AsDiscrete, Spacing
import nibabel as nib
import imageio # For creating GIFs
# --------------------------------

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Paths & Device ---
fracture_model_path = "/app/app/services/models/model.pt"
# --- Path to the downloaded MONAI Bundle (INSIDE Docker container) ---
if os.path.exists("/app/bundles"):  # Docker environment
    MONAI_BUNDLE_PATH = "/app/bundles/wholeBody_ct_segmentation"
else:  # Local development
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MONAI_BUNDLE_PATH = str(PROJECT_ROOT / "bundles" / "wholeBody_ct_segmentation")
device = torch.device("cpu")

# --- CheXNet Model & Transforms ---
chest_model = None
chest_transform = None
chest_model_target_layer = None #  Added target layer

def load_chest_model():
    """Loads the CheXNet model, transforms, and target layer on the first call."""
    global chest_model, chest_transform, chest_model_target_layer
    if chest_model is None:
        try:
            logger.info("Loading pre-trained CheXNet model (densenet121-res224-all)...")
            chest_model = xrv.models.DenseNet(weights="densenet121-res224-all")
            chest_model.eval()

            # ---Define the target layer for Grad-CAM ---
            chest_model_target_layer = [chest_model.features[-1]]
            logger.info("CheXNet model and target layer loaded successfully.")

        except Exception as e:
            # --- AUTO-FIX FOR CORRUPTED CACHE ---
            logger.warning(f"Initial load failed: {e}. Checking for corrupted cache...")
            try:
                # Try to locate and delete the corrupted file
                cache_dir = os.path.expanduser("~/.torchxrayvision/models_data")
                weight_file = os.path.join(cache_dir, "densenet121-res224-all.pt")

                if os.path.exists(weight_file):
                    logger.info(f"Removing corrupted cache file: {weight_file}")
                    os.remove(weight_file)

                logger.info("Retrying download and load...")
                chest_model = xrv.models.DenseNet(weights="densenet121-res224-all")
                chest_model.eval()
                chest_model_target_layer = [chest_model.features[-1]]
                logger.info("CheXNet model loaded successfully after retry.")

            except Exception as e2:
                logger.error(f"CRITICAL ERROR: Failed to load CheXNet model after retry: {e2}")
                raise RuntimeError(f"Failed to load AI model (CheXNet): {e2}")

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
# --- *** FIXED CHEST X-RAY ANALYSIS FUNCTION *** ---
# ---
def analyze_chest_xray(image_bytes: bytes) -> dict:
    """
    Analyzes chest x-ray for pathologies AND generates a Grad-CAM heatmap.
    FIXED: Now uses correct normalization range (-1024 to 1024) expected by TorchXRayVision.
    """
    if chest_model is None:
        load_chest_model()
        if chest_model is None:
             raise HTTPException(status_code=503, detail="CheXNet model is temporarily unavailable.")

    try:
        # 1. Load Image
        image = Image.open(io.BytesIO(image_bytes)).convert('L') # Grayscale

        # 2. Geometric Transform (Resize & Crop)
        # We apply this first so both Visualization and Tensor get the exact same crop
        geom_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224)
        ])
        image_cropped = geom_transform(image)

        # Convert to numpy for further processing
        image_np = np.asarray(image_cropped)

        # 3. Normalization for TorchXRayVision (CRITICAL FIX)
        # XRV expects input values roughly in range [-1024, 1024]
        # Previous code used ToTensor() which gave [0, 1], causing the model to see "black"
        img_norm = xrv.datasets.normalize(image_np, 255) # Scales 0-255 to -1024..1024

        # Add channel dimension if missing: (224, 224) -> (1, 224, 224)
        if img_norm.ndim == 2:
            img_norm = img_norm[None, ...]

        # Create Tensor
        img_tensor = torch.from_numpy(img_norm).float()
        img_tensor = img_tensor.unsqueeze(0) # Add batch dimension -> (1, 1, 224, 224)
        img_tensor.requires_grad_(True)

        # 4. Prepare for Visualization (GradCAM overlay)
        # We need a standard float RGB image [0, 1]
        vis_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        vis_rgb = np.float32(vis_rgb) / 255.0

        # 5. Get Model Predictions
        outputs = chest_model(img_tensor)
        probabilities = torch.sigmoid(outputs).cpu()

        # --- Find the pathology with the highest probability ---
        highest_prob = torch.max(probabilities)
        highest_prob_index = torch.argmax(probabilities)
        highest_pathology_name = chest_model.pathologies[highest_prob_index]

        logger.info(f"CheXNet prediction obtained. Highest prob: {highest_pathology_name} ({highest_prob:.3f})")

        # --- Generate Grad-CAM ---
        cam = GradCAM(model=chest_model, target_layers=chest_model_target_layer)
        targets = [ClassifierOutputTarget(highest_prob_index.item())]

        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # --- Create overlay ---
        # vis_rgb matches geometry perfectly now
        cam_image_overlay = show_cam_on_image(vis_rgb, grayscale_cam, use_rgb=True, image_weight=0.6)
        heatmap_image = Image.fromarray(cam_image_overlay)
        heatmap_base64 = convert_to_base64(heatmap_image)
        logger.info("CheXNet heatmap successfully generated.")

        # --- 6. Format results ---
        results = {}
        probabilities_np = probabilities.detach().numpy()[0]

        # Only return pathologies with probability > threshold
        threshold = 0.5

        for i, pathology in enumerate(chest_model.pathologies):
            prob = float(probabilities_np[i])
            if prob >= threshold:
                results[pathology] = round(prob, 3)

        # If nothing is above threshold, return the highest one or a "healthy" message
        if not results:
             results["Top Finding"] = f"{highest_pathology_name} ({round(float(highest_prob), 3)})"
             if float(highest_prob) < 0.5:
                 results["Status"] = "No significant pathologies detected"

        results["heatmap_base64"] = heatmap_base64
        results["heatmap_target"] = highest_pathology_name

        return {"analysis_results": results}

    except Exception as e:
        logger.exception(f"Error during CheXNet image analysis: {e}")
        raise HTTPException(status_code=500, detail=f"AI model error: {e}")


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

# --- UPDATED: Initialize global variables to None ---
bundle_config = None
vis_loader = None
OUTPUT_MASK_KEY = "pred" # Set a default

# --- NEW: Lazy loading function for config AND model ---
def load_brain_ct_resources():
    """Loads the Brain CT bundle config and visualization loader."""
    global bundle_config, vis_loader, OUTPUT_MASK_KEY

    # This function is called *after* test script overrides MONAI_BUNDLE_PATH
    if bundle_config is None:
        try:
            logger.info(f"Loading MONAI Bundle config from: {MONAI_BUNDLE_PATH}")

            # --- ***** THE FIX IS HERE ***** ---
            # We initialize the parser *with* the config file path
            config_file_path = f"{MONAI_BUNDLE_PATH}/configs/inference.json"
            bundle_config = ConfigParser()
            bundle_config.read_config(config_file_path)
            # --- ***** WE DO NOT CALL .read() ***** ---

            # Get the name of the output mask from the config
            try:
                # We access keys like a dictionary
                OUTPUT_MASK_KEY = bundle_config.get_parsed_content("handlers")[1].get("output_key", "pred")
            except Exception:
                OUTPUT_MASK_KEY = "pred" # Default for TotalSegmentator
                logger.warning(f"Could not parse 'output_key' from config, defaulting to '{OUTPUT_MASK_KEY}'")

            logger.info(f"MONAI Bundle config loaded. Output key: {OUTPUT_MASK_KEY}")

            # We also initialize the loader here
            vis_loader = Compose([
                LoadImage(image_only=True, ensure_channel_first=False), # Load as HxWxD
                Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                ScaleIntensity() # Scales to [0, 1]
            ])
            logger.info("MONAI visualization loader configured.")

        except Exception as e:
            logger.error(f"CRITICAL: Failed to load MONAI Bundle config: {e}")
            bundle_config = None # Ensure it's None on failure
            raise RuntimeError(f"Failed to load MONAI config: {e}")

def analyze_whole_body_ct_3d(temp_file_path: str) -> dict:
    """
    Analyzes a 3D NIfTI file using the 'wholeBody_ct_segmentation' bundle
    and returns an animated GIF of the segmentation overlay.
    """
    if bundle_config is None or vis_loader is None:
        load_brain_ct_resources()

    if bundle_config is None or vis_loader is None:
        raise HTTPException(status_code=503, detail="MONAI Bundle config or loader is not loaded.")

    inferer = None

    try:
        logger.info("Starting 3D analysis with MONAI Bundle (wholeBody_ct_segmentation)...")

        from monai.bundle import ConfigWorkflow

        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_str}")

        # === CRITICALLY IMPORTANT: Override the datalist ===
        inferer = ConfigWorkflow(
            config_file=f"{MONAI_BUNDLE_PATH}/configs/inference.json",
            workflow_type="inference",
            bundle_root=MONAI_BUNDLE_PATH,
            device=device_str,
            # Override datalist - pass our file
            datalist=[temp_file_path]
        )

        logger.info("Initializing MONAI workflow...")
        inferer.initialize()

        logger.info("Running 3D model inference...")
        inferer.run()

        logger.info("3D model inference complete.")

        # === Getting the result ===
        output_mask_tensor = None

        # Method 1: Via evaluator
        try:
            if hasattr(inferer, 'evaluator') and inferer.evaluator is not None:
                if hasattr(inferer.evaluator, 'state') and inferer.evaluator.state is not None:
                    output_data = inferer.evaluator.state.output

                    if output_data is not None:
                        logger.info(f"Output data keys: {list(output_data.keys())}")

                        if OUTPUT_MASK_KEY in output_data:
                            output_mask_tensor = output_data[OUTPUT_MASK_KEY]
                            logger.info(f"Found output using key '{OUTPUT_MASK_KEY}'")
                        else:
                            for key, val in output_data.items():
                                if isinstance(val, torch.Tensor):
                                    output_mask_tensor = val
                                    logger.info(f"Using first tensor with key: {key}")
                                    break
                    else:
                        logger.warning("evaluator.state.output is None")
                else:
                    logger.warning("evaluator.state is None or doesn't exist")
            else:
                logger.warning("evaluator is None or doesn't exist")
        except Exception as e:
            logger.warning(f"Failed to get output from evaluator.state: {e}")

        # Method 2: Via saved files
        if output_mask_tensor is None:
            logger.info("Attempting to load output from saved files...")
            try:
                output_dir = f"{MONAI_BUNDLE_PATH}/eval"
                if os.path.exists(output_dir):
                    import glob
                    files = glob.glob(f"{output_dir}/**/*trans.nii.gz", recursive=True)
                    if files:
                        latest_file = max(files, key=os.path.getctime)
                        logger.info(f"Loading output from: {latest_file}")

                        from monai.transforms import LoadImage
                        loader = LoadImage(image_only=True)
                        output_mask_tensor = loader(latest_file)

                        if isinstance(output_mask_tensor, torch.Tensor):
                            output_mask_tensor = output_mask_tensor.squeeze()
                            logger.info(f"Loaded output from file with shape: {output_mask_tensor.shape}")
                    else:
                        logger.warning(f"No output files found in {output_dir}")
                else:
                    logger.warning(f"Output directory doesn't exist: {output_dir}")
            except Exception as e:
                logger.warning(f"Failed to load output from files: {e}")

        if output_mask_tensor is None:
            raise RuntimeError(
                "Could not retrieve output mask from inference results. "
                "Check that the model ran successfully and produced outputs."
            )

        # === Dimension handling ===
        logger.info(f"Output mask shape: {output_mask_tensor.shape}, dtype: {output_mask_tensor.dtype}")

        if isinstance(output_mask_tensor, torch.Tensor):
            if output_mask_tensor.dim() == 5:  # [B, C, H, W, D]
                output_mask_tensor = torch.argmax(output_mask_tensor, dim=1).squeeze(0)
            elif output_mask_tensor.dim() == 4:  # [C, H, W, D]
                if output_mask_tensor.shape[0] > 1:
                    output_mask_tensor = torch.argmax(output_mask_tensor, dim=0)
                else:
                    output_mask_tensor = output_mask_tensor.squeeze(0)

            output_mask_tensor = output_mask_tensor.cpu().numpy()
        else:
            output_mask_tensor = np.array(output_mask_tensor)

        # === GIF generation ===
        logger.info("Generating animated GIF from 3D volume...")

        original_tensor = vis_loader(temp_file_path)
        logger.info(f"Original shape: {original_tensor.shape}, Mask shape: {output_mask_tensor.shape}")

        if original_tensor.shape != output_mask_tensor.shape:
            from monai.transforms import Resize
            resizer = Resize(spatial_size=original_tensor.shape, mode="nearest")
            output_mask_tensor = resizer(torch.tensor(output_mask_tensor).unsqueeze(0)).squeeze(0).numpy()
            logger.info(f"Resized mask to: {output_mask_tensor.shape}")

        frames = []
        gif_buffer = io.BytesIO()

        slice_indices = list(range(0, original_tensor.shape[2], max(1, original_tensor.shape[2] // 50)))
        logger.info(f"Creating GIF from {len(slice_indices)} slices...")

        for slice_idx in slice_indices:
            original_slice_np = original_tensor[:, :, slice_idx]
            mask_slice_np = (output_mask_tensor[:, :, slice_idx] > 0).astype(np.float32)
            overlay_frame = create_overlay_frame(original_slice_np, mask_slice_np)
            frames.append(overlay_frame)

        if len(frames) == 0:
            raise RuntimeError("No frames generated for GIF")

        imageio.mimsave(gif_buffer, frames, format='GIF', duration=0.1, loop=0)
        gif_bytes = gif_buffer.getvalue()
        heatmap_base64 = base64.b64encode(gif_bytes).decode("utf-8")
        logger.info(f"Animated GIF ({len(frames)} frames, {len(gif_bytes)} bytes) generated.")

        num_segmented_voxels = int(np.sum(output_mask_tensor > 0))
        total_voxels = int(np.prod(output_mask_tensor.shape))
        segmentation_percentage = (num_segmented_voxels / total_voxels) * 100

        finding = (
            f"3D Whole Body CT Segmentation completed successfully. "
            f"Segmented {num_segmented_voxels:,} voxels ({segmentation_percentage:.2f}% of volume). "
            f"Generated {len(frames)} frame animation."
        )

        return {
            "analysis_results": {
                "finding": finding,
                "model_used": "wholeBody_ct_segmentation",
                "heatmap_base64": heatmap_base64,
                "heatmap_target": "Animated Segmentation Mask (GIF)",
                "segmentation_stats": {
                    "segmented_voxels": num_segmented_voxels,
                    "total_voxels": total_voxels,
                    "percentage": round(segmentation_percentage, 2)
                }
            }
        }

    except Exception as e:
        logger.exception(f"Error during MONAI Bundle analysis: {e}")
        raise HTTPException(status_code=500, detail=f"AI model processing error (MONAI): {e}")
    finally:
        if inferer:
            try:
                inferer.finalize()
                logger.info("MONAI inferer finalized.")
            except Exception as e:
                logger.warning(f"Error during inferer finalization: {e}")
# --- Helper function for GIF creation (moved from inside analyze_whole_body_ct_3d) ---
# ---
def create_overlay_frame(original_slice, mask_slice):
    """Creates a single 2D overlay frame (CV2 Numpy array)"""
    # Normalize original slice to 8-bit [0, 255]
    min_val, max_val = np.min(original_slice), np.max(original_slice)
    denom = max_val - min_val
    if denom == 0: denom = 1.0 # Avoid division by zero

    original_slice_np = ((original_slice - min_val) / denom) * 255
    original_slice_np = original_slice_np.astype(np.uint8)
    vis_rgb = cv2.cvtColor(original_slice_np, cv2.COLOR_GRAY2RGB)

    # Normalize mask [0, 1] to [0, 255]
    mask_slice_np = (mask_slice * 255).astype(np.uint8)

    # Apply a color map (e.g., JET or HOT) to the grayscale mask
    color_heatmap = cv2.applyColorMap(mask_slice_np, cv2.COLORMAP_JET)

    # Make the heatmap transparent where mask is 0
    color_heatmap[mask_slice_np == 0] = 0

    # Blend the original image with the color heatmap
    overlay = cv2.addWeighted(vis_rgb, 0.7, color_heatmap, 0.3, 0)
    return overlay