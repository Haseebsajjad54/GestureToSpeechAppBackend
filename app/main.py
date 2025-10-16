from fastapi import FastAPI, Request
from app.schemas import PredictionRequest, PredictionResponse
from app.model import predict
import tensorflow as tf
import joblib
import os
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading model and label encoder...")
    MODEL_PATH = os.path.join("model", "rnn_model.h5")
    ENCODER_PATH = os.path.join("model", "rnn_label_encoder.joblib")

    app.state.model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        app.state.label_encoder = joblib.load(f)

    print("✅ Model and encoder loaded successfully.")
    yield
    print("🛑 Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "ok", "message": "Gesture-to-Speech backend is live!"}

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: Request, body: PredictionRequest):
    model = request.app.state.model
    label_encoder = request.app.state.label_encoder

    gesture, confidence = predict(model, label_encoder, body.features)
    return PredictionResponse(gesture=gesture, confidence=confidence)
