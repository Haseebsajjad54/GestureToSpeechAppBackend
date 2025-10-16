import numpy as np

def predict(features: list, model, label_encoder):
    # Convert input to array
    input_data = np.array(features).reshape(1, 20, 22)  # adjust shape as needed

    # Get prediction probabilities
    preds = model.predict(input_data)
    class_index = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    # Decode label
    gesture = label_encoder.inverse_transform([class_index])[0]
    print(gesture)
    return gesture, confidence

