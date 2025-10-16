from fastapi import FastAPI
import tensorflow as tf
import joblib
import os
from contextlib import asynccontextmanager

from app.schemas import PredictionRequest, PredictionResponse
from app.model import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup section ----
    print("🚀 Loading model and label encoder...")

    MODEL_PATH = os.path.join("model", "rnn_model.h5")
    ENCODER_PATH = os.path.join("model", "rnn_label_encoder.joblib")

    # Load model and encoder
    app.state.model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        app.state.label_encoder = joblib.load(f)

    print("✅ Model and encoder loaded successfully.")

    # Hand control back to FastAPI
    yield

    # ---- Shutdown section ----
    print("🧹 Releasing model and encoder...")
    del app.state.model
    del app.state.label_encoder
    print("✅ Resources released.")


app = FastAPI(
    title="Gesture Recognition API",
    lifespan=lifespan
)


@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "ok", "message": "Gesture-to-Speech backend is live!"}


@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: PredictionRequest):
    # Access model and encoder from app state
    model = request.app.state.model
    label_encoder = request.app.state.label_encoder

    gesture, confidence = predict(request.features, model, label_encoder)
    return PredictionResponse(gesture=gesture, confidence=confidence)
