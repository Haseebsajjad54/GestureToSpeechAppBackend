from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    features: List[List[float]]   # Example input features

class PredictionResponse(BaseModel):
    gesture: str
    confidence: float
