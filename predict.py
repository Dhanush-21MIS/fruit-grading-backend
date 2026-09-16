import os

# Render 512 MB CPU optimization: limit native thread pools before
# importing PyTorch so they do not allocate unnecessary worker memory.
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
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Your GitHub repository currently contains "Models" with capital M.
# This also supports "models" if your folder is lowercase.
if (BASE_DIR / "Models").exists():
    MODELS_DIR = BASE_DIR / "Models"
elif (BASE_DIR / "models").exists():
    MODELS_DIR = BASE_DIR / "models"
else:
    MODELS_DIR = BASE_DIR / "models"


IMG_SIZE = 224

QUALITY_CLASSES = [
    "good",
    "bad",
    "mixed"
]

# Your dataset contains these 8 fruit classes.
FRUIT_CLASSES = [
    "Apple",
    "Banana",
    "Grape",
    "Guava",
    "Lime",
    "Mango",
    "Orange",
    "Pomegranate"
]

MODEL_PATHS = {
    "EfficientNet": MODELS_DIR / "efficientnet_model.pth",
    "ConvNeXt": MODELS_DIR / "convnext_model.pth",
    "Swin": MODELS_DIR / "swin_model.pth"
}

MODEL_ARCHITECTURES = {
    "EfficientNet": "efficientnet_b0",
    "ConvNeXt": "convnext_tiny",
    "Swin": "swin_tiny_patch4_window7_224"
}


# ============================================================
# PREDICTION SETTINGS
# ============================================================

ENTROPY_THRESHOLD = 2.2

# Confidence is stored as percentage.
MIN_CONFIDENCE = 50.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Keep CPU inference memory predictable on Render.
try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

# MKLDNN may consume additional memory on a 512 MB service.
try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass

print("Using device:", device)
print("Models directory:", MODELS_DIR)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
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
            num_classes=0
        )

        feat_dim = self.backbone.num_features

        # Fruit classification head
        self.fruit_head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(
                feat_dim,
                len(FRUIT_CLASSES)
            )
        )

        # Quality classification head
        self.quality_head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(
                feat_dim,
                len(QUALITY_CLASSES)
            )
        )

    def forward(self, x):

        features = self.backbone(x)

        fruit_logits = self.fruit_head(features)

        quality_logits = self.quality_head(features)

        return fruit_logits, quality_logits


# ============================================================
# LOAD STATE DICT
# ============================================================

def load_state_dict(model_path):

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    print(f"Loading weights: {model_path}")

    # The model files are trusted files created by you.
    # weights_only=False is required for checkpoints saved
    # in the format used by your trained models.
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    # Handle common checkpoint formats
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    return cleaned_state_dict


# ============================================================
# BUILD ONE MODEL
# ============================================================

def build_model(model_name):
    """
    Build and load exactly ONE model.

    We intentionally do NOT use meta/to_empty here. The previous meta-based
    implementation caused invalid/uninitialized buffers in Swin.

    Instead:
      1. Build the real model normally.
      2. Memory-map the checkpoint where supported.
      3. Load the state_dict with assign=True.
      4. Run inference.
      5. Release the model before the next model is loaded.

    PyTorch documents mmap=True + assign=True as a memory-efficient
    checkpoint-loading approach.
    """

    if model_name not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    print("----------------------------------------")
    print(f"Loading {model_name} model...")
    print(f"Architecture: {MODEL_ARCHITECTURES[model_name]}")
    print(f"Path: {model_path}")
    print(
        f"Checkpoint size: "
        f"{model_path.stat().st_size / (1024 * 1024):.1f} MB"
    )

    model = None
    state_dict = None

    try:
        # Normal construction is intentional. It initializes architecture
        # buffers correctly, including Swin's attention/index buffers.
        model = MultiTaskModel(
            MODEL_ARCHITECTURES[model_name]
        )

        # Load the trusted checkpoint with mmap when available.
        try:
            state_dict = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except (TypeError, RuntimeError, ValueError, OSError):
            # Fallback for legacy/unsupported checkpoint serialization.
            state_dict = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
            )

        # Support the checkpoint formats used by the project.
        if isinstance(state_dict, dict):

            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Invalid checkpoint format for {model_name}."
            )

        # Remove DataParallel prefix if present.
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):
                key = key[7:]

            cleaned_state_dict[key] = value

        # The outer checkpoint dictionary can now be released.
        del state_dict
        state_dict = cleaned_state_dict
        del cleaned_state_dict

        gc.collect()

        # assign=True avoids copying every checkpoint tensor into a second
        # tensor allocation. Unlike the previous version, the model here is
        # a real CPU model, NOT a meta model.
        try:
            model.load_state_dict(
                state_dict,
                strict=True,
                assign=True,
            )
        except TypeError:
            # Compatibility fallback for older PyTorch versions.
            model.load_state_dict(
                state_dict,
                strict=True,
            )

        # Release checkpoint references immediately.
        del state_dict
        state_dict = None
        gc.collect()

        model.to(device)
        model.eval()

        print(f"{model_name} loaded successfully.")

        return model

    except Exception:

        if state_dict is not None:
            del state_dict

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

        raise


# ============================================================
# PREDICT USING ONE MODEL
# ============================================================

def predict_single_model(model, image_tensor):

    start_time = time.time()

    with torch.inference_mode():

        fruit_logits, quality_logits = model(
            image_tensor
        )

        # Fruit probabilities
        fruit_probs = torch.softmax(
            fruit_logits,
            dim=1
        )

        # Quality probabilities
        quality_probs = torch.softmax(
            quality_logits,
            dim=1
        )

        # Fruit prediction
        fruit_probabilities = fruit_probs[0]

        fruit_confidence, fruit_index = torch.max(
            fruit_probabilities,
            dim=0
        )

        fruit_label = FRUIT_CLASSES[
            fruit_index.item()
        ]

        # Quality prediction
        quality_confidence, quality_index = torch.max(
            quality_probs[0],
            dim=0
        )

        quality_label = QUALITY_CLASSES[
            quality_index.item()
        ]

        # Entropy
        entropy = -torch.sum(
            fruit_probabilities *
            torch.log(fruit_probabilities + 1e-10)
        ).item()

    end_time = time.time()

    return {
        "fruit": fruit_label,

        "fruit_confidence":
            round(
                fruit_confidence.item() * 100,
                2
            ),

        "quality": quality_label,

        "quality_confidence":
            round(
                quality_confidence.item() * 100,
                2
            ),

        "entropy":
            round(entropy, 4),

        "time_taken":
            round(
                end_time - start_time,
                4
            ),

        # Keep probability tensors for ensemble
        "_fruit_probs": fruit_probs.cpu(),

        "_quality_probs": quality_probs.cpu()
    }


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_model(model):

    if model is not None:

        model.cpu()

        del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(image_tensor):

    total_start_time = time.time()

    individual_predictions = {}

    fruit_probability_list = []

    quality_probability_list = []

    best_model_name = None

    best_combined_confidence = -1


    # ========================================================
    # IMPORTANT:
    # Only ONE model exists in memory at a time.
    # ========================================================

    for model_name in [
        "EfficientNet",
        "ConvNeXt",
        "Swin"
    ]:

        model = None

        try:

            # Load one model
            model = build_model(model_name)

            # Predict
            result = predict_single_model(
                model,
                image_tensor
            )

            # Store normal prediction information
            individual_predictions[model_name] = {
                "fruit": result["fruit"],

                "fruit_confidence":
                    result["fruit_confidence"],

                "quality": result["quality"],

                "quality_confidence":
                    result["quality_confidence"]
            }

            # Save probability tensors
            fruit_probability_list.append(
                result["_fruit_probs"]
            )

            quality_probability_list.append(
                result["_quality_probs"]
            )

            # Find best model
            combined_confidence = (
                result["fruit_confidence"]
                +
                result["quality_confidence"]
            ) / 2

            if (
                combined_confidence
                >
                best_combined_confidence
            ):

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

            # VERY IMPORTANT FOR RENDER RAM
            cleanup_model(model)

            model = None

            print(
                f"{model_name} released from memory."
            )


    # ========================================================
    # ENSEMBLE
    # ========================================================

    avg_fruit_probs = torch.mean(
        torch.stack(fruit_probability_list),
        dim=0
    )

    avg_quality_probs = torch.mean(
        torch.stack(quality_probability_list),
        dim=0
    )


    # ========================================================
    # Final fruit prediction
    # ========================================================

    fruit_probs = avg_fruit_probs[0]

    fruit_confidence, fruit_index = torch.max(
        fruit_probs,
        dim=0
    )

    ensemble_fruit = FRUIT_CLASSES[
        fruit_index.item()
    ]


    # ========================================================
    # Final quality prediction
    # ========================================================

    quality_probs = avg_quality_probs[0]

    quality_confidence, quality_index = torch.max(
        quality_probs,
        dim=0
    )

    ensemble_quality = QUALITY_CLASSES[
        quality_index.item()
    ]


    # ========================================================
    # Entropy
    # ========================================================

    entropy = -torch.sum(
        fruit_probs *
        torch.log(fruit_probs + 1e-10)
    ).item()


    # ========================================================
    # Model disagreement
    # ========================================================

    fruit_predictions = [
        individual_predictions[name]["fruit"]
        for name in individual_predictions
    ]

    unique_predictions = set(
        fruit_predictions
    )

    model_disagreement = (
        len(unique_predictions) > 1
    )


    # ========================================================
    # Final filtering
    # ========================================================

    ensemble_fruit_confidence = (
        fruit_confidence.item() * 100
    )

    if (
        entropy > ENTROPY_THRESHOLD
        or
        ensemble_fruit_confidence < MIN_CONFIDENCE
        or
        model_disagreement
    ):

        final_fruit = "Unknown Fruit"

    else:

        final_fruit = ensemble_fruit


    # ========================================================
    # Total processing time
    # ========================================================

    total_time = (
        time.time() - total_start_time
    )


    # ========================================================
    # Final response
    # ========================================================

    result = {

        "fruit": final_fruit,

        "predicted_fruit_before_filter":
            ensemble_fruit,

        "fruit_confidence":
            round(
                ensemble_fruit_confidence,
                2
            ),

        "quality":
            ensemble_quality,

        "quality_confidence":
            round(
                quality_confidence.item() * 100,
                2
            ),

        "entropy":
            round(
                entropy,
                4
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
                4
            ),

        "individual_predictions":
            individual_predictions
    }


    print("----------------------------------------")
    print("ENSEMBLE RESULT")
    print("----------------------------------------")

    print(
        "Fruit:",
        final_fruit
    )

    print(
        "Fruit before filter:",
        ensemble_fruit
    )

    print(
        "Fruit confidence:",
        round(
            ensemble_fruit_confidence,
            2
        ),
        "%"
    )

    print(
        "Quality:",
        ensemble_quality
    )

    print(
        "Quality confidence:",
        round(
            quality_confidence.item() * 100,
            2
        ),
        "%"
    )

    print(
        "Entropy:",
        round(
            entropy,
            4
        )
    )

    print(
        "Best model:",
        best_model_name
    )

    print(
        "Model disagreement:",
        model_disagreement
    )

    print(
        "Total time:",
        round(
            total_time,
            4
        ),
        "seconds"
    )

    print("----------------------------------------")


    return result


# ============================================================
# PUBLIC API FUNCTION
# ============================================================

def predict_image(image: Image.Image):

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )


    # Convert image to RGB
    image = image.convert("RGB")


    # Transform
    image_tensor = transform(
        image
    ).unsqueeze(0)


    # Move input to device
    image_tensor = image_tensor.to(
        device
    )


    # Ensemble prediction
    result = ensemble_predict(
        image_tensor
    )


    # Cleanup input tensor
    del image_tensor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return result