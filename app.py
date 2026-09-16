from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from predict import predict_image


app = FastAPI(
    title="Fruit Grading API",
    description="Fruit classification and quality grading using EfficientNet, ConvNeXt and Swin Transformer.",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Fruit Grading API is running",
        "docs": "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:
        # Read uploaded image
        image_data = await file.read()

        # Convert uploaded image to RGB PIL image
        image = Image.open(
            BytesIO(image_data)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read image: {str(e)}"
        )

    try:
        # Run ML prediction
        result = predict_image(image)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )