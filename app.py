from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from predict import predict_image


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Fruit Grading API",
    description=(
        "Fruit classification and quality grading using "
        "EfficientNet, ConvNeXt and Swin Transformer."
    ),
    version="1.0.0",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        # Local Vite development
        "http://localhost:5173",
        "http://127.0.0.1:5173",

        # Production Render frontend
        "https://fruit-grading-ui.onrender.com",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Fruit Grading API is running",
        "docs": "/docs",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate uploaded file
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        image_data = await file.read()

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        image = Image.open(
            BytesIO(image_data)
        ).convert("RGB")

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read image: {str(e)}",
        )

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    try:

        print(
            f"Prediction request received: "
            f"{file.filename}"
        )

        result = predict_image(image)

        print(
            f"Prediction completed: "
            f"{result.get('fruit')}"
        )

        return result

    except Exception as e:

        print(
            "Prediction error:",
            repr(e),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )