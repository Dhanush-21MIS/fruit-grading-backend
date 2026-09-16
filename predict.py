# predict.py
# Low-memory inference version for Render 512 MB CPU service.
#
# Important:
# - Models are loaded ONE AT A TIME.
# - There is NO global model cache.
# - Checkpoints are released immediately after loading.
# - CPU threads are limited before importing torch.
# - The public predict_image(image) API is preserved.

import os

# Limit native CPU thread memory before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import time
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms
from torch import nn


# ============================================================
# LOW-MEMORY PYTORCH SETTINGS
# ============================================================

try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

# MKLDNN can use additional CPU memory. Disabling it is slower,
# but is useful on a 512 MB Render instance.
try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Render build downloads model files into lowercase "models".
# The capitalized fallback keeps local Windows execution compatible.
if (BASE_DIR / "models").exists():
    MODELS_DIR = BASE_DIR / "models"
elif (BASE_DIR / "Models").exists():
    MODELS_DIR = BASE_DIR / "Models"
else:
    MODELS_DIR = BASE_DIR / "models"

IMG_SIZE = 224

QUALITY_CLASSES = [
    "good",
    "bad",
    "mixed",
]

FRUIT_CLASSES = [
    "Apple",
    "Banana",
    "Grape",
    "Guava",
    "Lime",
    "Mango",
    "Orange",
    "Pomegranate",
]

MODEL_PATHS = {
    "EfficientNet": MODELS_DIR / "efficientnet_model.pth",
    "ConvNeXt": MODELS_DIR / "convnext_model.pth",
    "Swin": MODELS_DIR / "swin_model.pth",
}

MODEL_ARCHITECTURES = {
    "EfficientNet": "efficientnet_b0",
    "ConvNeXt": "convnext_tiny",
    "Swin": "swin_tiny_patch4_window7_224",
}

# Existing project filtering rules.
ENTROPY_THRESHOLD = 2.2
MIN_CONFIDENCE = 50.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("Models directory:", MODELS_DIR)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# MULTI-TASK MODEL
# ============================================================

class MultiTaskModel(nn.Module):

    def __init__(self, backbone_name):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
        )

        feat_dim = self.backbone.num_features

        self.fruit_head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, len(FRUIT_CLASSES)),
        )

        self.quality_head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, len(QUALITY_CLASSES)),
        )

    def forward(self, x):
        features = self.backbone(x)

        fruit_logits = self.fruit_head(features)
        quality_logits = self.quality_head(features)

        return fruit_logits, quality_logits


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def _extract_state_dict(checkpoint):
    """Support the checkpoint formats used by the project."""

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")

    # Remove DataParallel's "module." prefix when present.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    return cleaned_state_dict


def load_state_dict(model_path):
    """
    Load a trusted project checkpoint.

    mmap=True reduces the amount of physical memory used while reading
    supported PyTorch zip checkpoints. A fallback is provided for older
    or legacy checkpoint formats.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    file_size_mb = model_path.stat().st_size / (1024 * 1024)

    print(
        f"Loading weights: {model_path} "
        f"({file_size_mb:.1f} MB)"
    )

    checkpoint = None

    try:
        # mmap lazily maps supported checkpoint storage instead of
        # immediately copying the entire file into RAM.
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

    except (TypeError, RuntimeError, ValueError, OSError):
        # Fallback for checkpoint formats that do not support mmap.
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

    state_dict = _extract_state_dict(checkpoint)

    # Release the outer checkpoint reference immediately.
    del checkpoint
    gc.collect()

    return state_dict


# ============================================================
# BUILD ONE MODEL ONLY
# ============================================================

def build_model(model_name):

    if model_name not in MODEL_ARCHITECTURES:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_PATHS[model_name]

    print("----------------------------------------")
    print(f"Loading {model_name} model...")
    print(f"Architecture: {MODEL_ARCHITECTURES[model_name]}")
    print(f"Path: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Expected directory: {MODELS_DIR}"
        )

    # Create ONLY the requested architecture.
    model = MultiTaskModel(
        MODEL_ARCHITECTURES[model_name]
    )

    # Load checkpoint after model creation. This avoids keeping a
    # second full model architecture in memory.
    state_dict = load_state_dict(model_path)

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    finally:
        # The model now owns its parameters. The temporary checkpoint
        # tensors are no longer needed.
        del state_dict
        gc.collect()

    model.to(device)
    model.eval()

    print(f"{model_name} loaded successfully.")

    return model


# ============================================================
# SINGLE-MODEL PREDICTION
# ============================================================

def predict_single_model(model, image_tensor):

    start_time = time.time()

    with torch.inference_mode():

        fruit_logits, quality_logits = model(image_tensor)

        fruit_probs = torch.softmax(
            fruit_logits,
            dim=1,
        )

        quality_probs = torch.softmax(
            quality_logits,
            dim=1,
        )

        fruit_probabilities = fruit_probs[0]

        fruit_confidence, fruit_index = torch.max(
            fruit_probabilities,
            dim=0,
        )

        fruit_label = FRUIT_CLASSES[
            fruit_index.item()
        ]

        quality_confidence, quality_index = torch.max(
            quality_probs[0],
            dim=0,
        )

        quality_label = QUALITY_CLASSES[
            quality_index.item()
        ]

        entropy = -torch.sum(
            fruit_probabilities
            * torch.log(fruit_probabilities + 1e-10)
        ).item()

        # Only these tiny vectors are retained for the ensemble.
        fruit_probs_cpu = fruit_probs.cpu()
        quality_probs_cpu = quality_probs.cpu()

    end_time = time.time()

    return {
        "fruit": fruit_label,

        "fruit_confidence": round(
            fruit_confidence.item() * 100,
            2,
        ),

        "quality": quality_label,

        "quality_confidence": round(
            quality_confidence.item() * 100,
            2,
        ),

        "entropy": round(entropy, 4),

        "time_taken": round(
            end_time - start_time,
            4,
        ),

        "_fruit_probs": fruit_probs_cpu,

        "_quality_probs": quality_probs_cpu,
    }


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_model(model):

    if model is not None:

        try:
            model.cpu()
        except Exception:
            pass

        del model

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def cleanup_memory():
    """General cleanup after a prediction request."""

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(image_tensor):

    total_start_time = time.time()

    individual_predictions = {}

    fruit_probability_list = []
    quality_probability_list = []

    best_model_name = None
    best_combined_confidence = -1.0

    model_names = [
        "EfficientNet",
        "ConvNeXt",
        "Swin",
    ]

    # IMPORTANT:
    # Only ONE neural network is loaded at a time.
    for model_name in model_names:

        model = None

        try:

            model = build_model(model_name)

            result = predict_single_model(
                model,
                image_tensor,
            )

            individual_predictions[model_name] = {
                "fruit": result["fruit"],

                "fruit_confidence":
                    result["fruit_confidence"],

                "quality": result["quality"],

                "quality_confidence":
                    result["quality_confidence"],
            }

            # Each item is only 8 or 3 probability values,
            # so retaining these does not meaningfully increase RAM.
            fruit_probability_list.append(
                result["_fruit_probs"]
            )

            quality_probability_list.append(
                result["_quality_probs"]
            )

            combined_confidence = (
                result["fruit_confidence"]
                + result["quality_confidence"]
            ) / 2.0

            if combined_confidence > best_combined_confidence:

                best_combined_confidence = (
                    combined_confidence
                )

                best_model_name = model_name

            print(
                f"{model_name}: "
                f"{result['fruit']} | "
                f"{result['fruit_confidence']}% | "
                f"{result['quality']} | "
                f"{result['quality_confidence']}%"
            )

        finally:

            # Release the entire neural network before loading
            # the next one.
            cleanup_model(model)

            model = None

            print(
                f"{model_name} released from memory."
            )

    if len(fruit_probability_list) != 3:
        raise RuntimeError(
            "Not all three models produced predictions."
        )

    # ========================================================
    # ENSEMBLE
    # ========================================================

    avg_fruit_probs = torch.mean(
        torch.stack(fruit_probability_list),
        dim=0,
    )

    avg_quality_probs = torch.mean(
        torch.stack(quality_probability_list),
        dim=0,
    )

    # Release the individual lists now that the averages exist.
    del fruit_probability_list
    del quality_probability_list

    # ========================================================
    # FINAL FRUIT
    # ========================================================

    fruit_probs = avg_fruit_probs[0]

    fruit_confidence, fruit_index = torch.max(
        fruit_probs,
        dim=0,
    )

    ensemble_fruit = FRUIT_CLASSES[
        fruit_index.item()
    ]

    # ========================================================
    # FINAL QUALITY
    # ========================================================

    quality_probs = avg_quality_probs[0]

    quality_confidence, quality_index = torch.max(
        quality_probs,
        dim=0,
    )

    ensemble_quality = QUALITY_CLASSES[
        quality_index.item()
    ]

    # ========================================================
    # ENTROPY
    # ========================================================

    entropy = -torch.sum(
        fruit_probs
        * torch.log(fruit_probs + 1e-10)
    ).item()

    # ========================================================
    # MODEL DISAGREEMENT
    # ========================================================

    fruit_predictions = [
        individual_predictions[name]["fruit"]
        for name in individual_predictions
    ]

    unique_predictions = set(fruit_predictions)

    model_disagreement = (
        len(unique_predictions) > 1
    )

    # ========================================================
    # FINAL FILTER
    # ========================================================

    ensemble_fruit_confidence = (
        fruit_confidence.item() * 100
    )

    if (
        entropy > ENTROPY_THRESHOLD
        or ensemble_fruit_confidence < MIN_CONFIDENCE
        or model_disagreement
    ):

        final_fruit = "Unknown Fruit"

    else:

        final_fruit = ensemble_fruit

    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_time = (
        time.time() - total_start_time
    )

    result = {

        "fruit":
            final_fruit,

        "predicted_fruit_before_filter":
            ensemble_fruit,

        "fruit_confidence":
            round(
                ensemble_fruit_confidence,
                2,
            ),

        "quality":
            ensemble_quality,

        "quality_confidence":
            round(
                quality_confidence.item() * 100,
                2,
            ),

        "entropy":
            round(
                entropy,
                4,
            ),

        "best_model":
            best_model_name,

        "model_disagreement":
            model_disagreement,

        "device":
            str(device),

        "time_taken_seconds":
            round(
                total_time,
                4,
            ),

        "individual_predictions":
            individual_predictions,
    }

    print("----------------------------------------")
    print("ENSEMBLE RESULT")
    print("----------------------------------------")
    print("Fruit:", final_fruit)
    print("Fruit before filter:", ensemble_fruit)
    print(
        "Fruit confidence:",
        round(ensemble_fruit_confidence, 2),
        "%",
    )
    print("Quality:", ensemble_quality)
    print(
        "Quality confidence:",
        round(
            quality_confidence.item() * 100,
            2,
        ),
        "%",
    )
    print("Entropy:", round(entropy, 4))
    print("Best model:", best_model_name)
    print("Model disagreement:", model_disagreement)
    print(
        "Total time:",
        round(total_time, 4),
        "seconds",
    )
    print("----------------------------------------")

    # Release ensemble tensors.
    del avg_fruit_probs
    del avg_quality_probs

    cleanup_memory()

    return result


# ============================================================
# PUBLIC API FUNCTION
# ============================================================

def predict_image(image: Image.Image):

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    try:

        image = image.convert("RGB")

        image_tensor = transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            device
        )

        result = ensemble_predict(
            image_tensor
        )

        return result

    finally:

        # Always release the request tensor even if prediction fails.
        try:
            del image_tensor
        except Exception:
            pass

        cleanup_memory()
