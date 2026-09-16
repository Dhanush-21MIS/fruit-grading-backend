import time
from pathlib import Path

import torch
import timm
from PIL import Image
from torch import nn
from torchvision import transforms


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# CLASSES
# ============================================================

# IMPORTANT:
# These must match the classes used during training.

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

QUALITY_CLASSES = [
    "good",
    "bad",
    "mixed"
]


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224

ENTROPY_THRESHOLD = 2.2

MIN_CONFIDENCE = 0.5


# ============================================================
# MODEL DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"


MODEL_PATHS = {

    "EfficientNet":
        MODEL_DIR / "efficientnet_model.pth",

    "ConvNeXt":
        MODEL_DIR / "convnext_model.pth",

    "Swin":
        MODEL_DIR / "swin_model.pth"
}


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor()
])


# ============================================================
# MULTI-TASK MODEL
# ============================================================

class MultiTaskModel(nn.Module):

    def __init__(
        self,
        backbone_name,
        dropout=0.4
    ):

        super().__init__()

        # Backbone
        self.backbone = timm.create_model(

            backbone_name,

            pretrained=False,

            num_classes=0
        )

        # Number of extracted features
        feat_dim = self.backbone.num_features


        # ----------------------------------------------------
        # Fruit classification head
        # ----------------------------------------------------

        self.fruit_head = nn.Sequential(

            nn.BatchNorm1d(
                feat_dim
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                feat_dim,
                len(FRUIT_CLASSES)
            )
        )


        # ----------------------------------------------------
        # Quality classification head
        # ----------------------------------------------------

        self.quality_head = nn.Sequential(

            nn.BatchNorm1d(
                feat_dim
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                feat_dim,
                len(QUALITY_CLASSES)
            )
        )


    def forward(self, x):

        features = self.backbone(x)

        fruit_output = self.fruit_head(
            features
        )

        quality_output = self.quality_head(
            features
        )

        return fruit_output, quality_output


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

def load_state_dict(path):

    try:

        return torch.load(
            path,
            map_location=device,
            weights_only=True
        )

    except TypeError:

        return torch.load(
            path,
            map_location=device
        )


# ============================================================
# BUILD ALL MODELS
# ============================================================

def build_models():

    configs = {

        "EfficientNet": (
            "efficientnet_b0",
            0.7
        ),

        "ConvNeXt": (
            "convnext_tiny",
            0.7
        ),

        "Swin": (
            "swin_tiny_patch4_window7_224",
            0.6
        )
    }


    loaded_models = {}


    for name, (
        backbone_name,
        dropout
    ) in configs.items():

        model_path = MODEL_PATHS[name]


        # Check model file
        if not model_path.exists():

            raise FileNotFoundError(

                f"Model file not found: {model_path}\n"
                f"Please place the trained {name} "
                f"model inside the models folder."
            )


        print(
            f"Loading {name} model..."
        )


        # Create architecture
        model = MultiTaskModel(

            backbone_name,

            dropout=dropout
        )


        # Load trained weights
        state_dict = load_state_dict(
            model_path
        )


        model.load_state_dict(
            state_dict
        )


        # Move to GPU/CPU
        model.to(device)


        # Evaluation mode
        model.eval()


        loaded_models[name] = model


        print(
            f"{name} loaded successfully."
        )


    return loaded_models


# ============================================================
# LOAD MODELS WHEN SERVER STARTS
# ============================================================

models = build_models()


# ============================================================
# SINGLE MODEL PREDICTION
# ============================================================

def single_model_prediction(
    model,
    image_tensor
):

    with torch.inference_mode():

        fruit_logits, quality_logits = model(
            image_tensor
        )


        # Fruit probabilities
        fruit_probs = torch.softmax(
            fruit_logits,
            dim=1
        )[0]


        # Quality probabilities
        quality_probs = torch.softmax(
            quality_logits,
            dim=1
        )[0]


        # Fruit prediction
        fruit_confidence, fruit_index = torch.max(
            fruit_probs,
            dim=0
        )


        # Quality prediction
        quality_confidence, quality_index = torch.max(
            quality_probs,
            dim=0
        )


        # Entropy
        entropy = -torch.sum(

            fruit_probs *

            torch.log(
                fruit_probs + 1e-10
            )

        ).item()


    return {

        "fruit":
            FRUIT_CLASSES[
                fruit_index.item()
            ],

        "fruit_confidence":
            float(
                fruit_confidence.item()
            ),

        "quality":
            QUALITY_CLASSES[
                quality_index.item()
            ],

        "quality_confidence":
            float(
                quality_confidence.item()
            ),

        "entropy":
            float(entropy)
    }


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_image(image: Image.Image):

    start_time = time.time()


    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image = image.convert("RGB")


    image_tensor = transform(
        image
    ).unsqueeze(0)


    image_tensor = image_tensor.to(
        device
    )


    # --------------------------------------------------------
    # Probability lists for ensemble
    # --------------------------------------------------------

    fruit_probs_list = []

    quality_probs_list = []


    # Individual model results
    individual_predictions = {}


    # Best model
    best_model_name = None

    best_combined_confidence = -1


    # ========================================================
    # RUN ALL THREE MODELS
    # ========================================================

    with torch.inference_mode():

        for name, model in models.items():

            fruit_logits, quality_logits = model(
                image_tensor
            )


            # ------------------------------------------------
            # Convert logits to probabilities
            # ------------------------------------------------

            fruit_probs = torch.softmax(
                fruit_logits,
                dim=1
            )


            quality_probs = torch.softmax(
                quality_logits,
                dim=1
            )


            # Store probabilities
            fruit_probs_list.append(
                fruit_probs
            )

            quality_probs_list.append(
                quality_probs
            )


            # ------------------------------------------------
            # Fruit confidence
            # ------------------------------------------------

            fruit_confidence = torch.max(
                fruit_probs[0]
            ).item()


            # ------------------------------------------------
            # Quality confidence
            # ------------------------------------------------

            quality_confidence = torch.max(
                quality_probs[0]
            ).item()


            # Combined confidence
            combined_confidence = (

                fruit_confidence +

                quality_confidence

            ) / 2


            # Determine best model
            if combined_confidence > best_combined_confidence:

                best_combined_confidence = (
                    combined_confidence
                )

                best_model_name = name


            # ------------------------------------------------
            # Individual predictions
            # ------------------------------------------------

            fruit_index = torch.argmax(
                fruit_probs[0]
            ).item()


            quality_index = torch.argmax(
                quality_probs[0]
            ).item()


            individual_predictions[name] = {

                "fruit":
                    FRUIT_CLASSES[
                        fruit_index
                    ],

                "fruit_confidence":
                    round(
                        fruit_confidence * 100,
                        2
                    ),

                "quality":
                    QUALITY_CLASSES[
                        quality_index
                    ],

                "quality_confidence":
                    round(
                        quality_confidence * 100,
                        2
                    )
            }


        # ====================================================
        # ENSEMBLE
        # ====================================================

        avg_fruit_probs = torch.mean(

            torch.stack(
                fruit_probs_list
            ),

            dim=0
        )[0]


        avg_quality_probs = torch.mean(

            torch.stack(
                quality_probs_list
            ),

            dim=0
        )[0]


        # ----------------------------------------------------
        # Ensemble fruit prediction
        # ----------------------------------------------------

        fruit_confidence, fruit_index = torch.max(

            avg_fruit_probs,

            dim=0
        )


        # ----------------------------------------------------
        # Ensemble quality prediction
        # ----------------------------------------------------

        quality_confidence, quality_index = torch.max(

            avg_quality_probs,

            dim=0
        )


        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------

        entropy = -torch.sum(

            avg_fruit_probs *

            torch.log(
                avg_fruit_probs + 1e-10
            )

        ).item()


    # ========================================================
    # LABELS
    # ========================================================

    ensemble_fruit = FRUIT_CLASSES[
        fruit_index.item()
    ]


    ensemble_quality = QUALITY_CLASSES[
        quality_index.item()
    ]


    # ========================================================
    # MODEL DISAGREEMENT
    # ========================================================

    unique_fruits = {

        prediction["fruit"]

        for prediction
        in individual_predictions.values()
    }


    disagreement = (
        len(unique_fruits) > 1
    )


    # ========================================================
    # UNKNOWN FRUIT FILTER
    # ========================================================

    if (

        entropy > ENTROPY_THRESHOLD

        or

        fruit_confidence.item()
        < MIN_CONFIDENCE

        or

        disagreement

    ):

        final_fruit = "Unknown Fruit"

    else:

        final_fruit = ensemble_fruit


    # ========================================================
    # FINAL RESULT
    # ========================================================

    end_time = time.time()


    return {

        "fruit":
            final_fruit,

        "predicted_fruit_before_filter":
            ensemble_fruit,

        "fruit_confidence":
            round(
                fruit_confidence.item() * 100,
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
            disagreement,

        "device":
            str(device),

        "time_taken_seconds":
            round(
                end_time - start_time,
                4
            ),

        "individual_predictions":
            individual_predictions
    }