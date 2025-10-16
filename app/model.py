import numpy as np

def predict(model, label_encoder, features: list):
    input_data = np.array(features).reshape(1, 20, 22)
    preds = model.predict(input_data)
    class_index = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    gesture = label_encoder.inverse_transform([class_index])[0]
    return gesture, confidence
