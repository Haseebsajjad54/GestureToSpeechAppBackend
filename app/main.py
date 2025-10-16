from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse
from app.model import predict

app = FastAPI(title="Gesture Recognition API")
@app.get("/")
def home():
    return {"message": "API is running successfully on Render!"}

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: PredictionRequest):
    gesture, confidence = predict(request.features)
    return PredictionResponse(gesture=gesture, confidence=confidence)
