from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse
from app.model import predict

app = FastAPI(title="Gesture Recognition API")
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "ok", "message": "Gesture-to-Speech backend is live!"}

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: PredictionRequest):
    gesture, confidence = predict(request.features)
    return PredictionResponse(gesture=gesture, confidence=confidence)
