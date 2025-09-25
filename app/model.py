import tensorflow as tf
import numpy as np
import joblib
import os

# Paths
MODEL_PATH = os.path.join("model", "rnn_model.h5")
ENCODER_PATH = os.path.join("model", "rnn_label_encoder.joblib")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load label encoder
with open(ENCODER_PATH, "rb") as f:
    label_encoder = joblib.load(f)

def predict(features: list):
    # Convert input to array
    input_data = np.array(features).reshape(1, 20, 22)  # batch=1

    # Get prediction probabilities
    preds = model.predict(input_data)
    class_index = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    # Decode label
    gesture = label_encoder.inverse_transform([class_index])[0]
    print(gesture)
    return gesture, confidence
