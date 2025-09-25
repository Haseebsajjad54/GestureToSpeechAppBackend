!pip install gTTS

from gtts import gTTS
from IPython.display import Audio, display
import tempfile

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Bidirectional, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import regularizers
import random

# === Step 1: Load & Clean Data ===
def load_and_preprocess_data(file_paths):
    dfs = [pd.read_csv(fp, encoding='utf-8') for fp in file_paths]
    df = pd.concat(dfs, ignore_index=True)

    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].astype(str).values
    return X, y

# === Step 2: Scale and Encode ===
def scale_and_encode(X, y, fit=False, scaler=None, label_encoder=None, one_hot_encoder=None):
    if fit:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        label_encoder = LabelEncoder()
        y_int = label_encoder.fit_transform(y)

        one_hot_encoder = OneHotEncoder(sparse_output=False)
        y_encoded = one_hot_encoder.fit_transform(y_int.reshape(-1, 1))
    else:
        X = scaler.transform(X)
        y_int = label_encoder.transform(y)
        y_encoded = one_hot_encoder.transform(y_int.reshape(-1, 1))

    X = X.reshape((X.shape[0], 20, 22))  # Reshape to (samples, time_steps, features)
    return X, y_encoded, scaler, label_encoder, one_hot_encoder

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.5),

        LSTM(64, activation='tanh'),
        Dropout(0.5),

        Dense(64, activation='swish', kernel_regularizer=regularizers.l2(0.05)),
        Dense(num_classes, activation='softmax')  # Final classification layer
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model

from tensorflow.keras.layers import Layer, Input, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Activation, Multiply, Permute, RepeatVector
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class TemporalAttention(Layer):
    def __init__(self):
        super(TemporalAttention, self).__init__()

    def call(self, inputs):

        attention = Dense(1, activation='tanh')(inputs)
        attention = K.squeeze(attention, axis=-1)
        attention_weights = Activation('softmax')(attention)
        attention_weights = K.expand_dims(attention_weights, axis=-1)
        weighted = inputs * attention_weights
        return K.sum(weighted, axis=1)

def build_lstm_attention_model(input_shape, num_classes):
    inp = Input(shape=input_shape)

    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = BatchNormalization()
    x = Dropout(0.5)(x)

    x = TemporalAttention()(x)

    x = Dense(64, activation='swish', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def speak_prediction_urdu(text):
    """Converts the given text to Urdu speech using gTTS and plays it."""
    try:
        tts = gTTS(text=text, lang='ur')  # 'ur' is the language code for Urdu
        tts.save("prediction.mp3")
        display(Audio("prediction.mp3", autoplay=True))
    except Exception as e:
        print(f"Error in speak_prediction_urdu: {e}")

# === Step 4: Train Model ===
def train_model(model, X_train, y_train, X_val, y_val):
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32,
              callbacks=[early_stop, lr_scheduler])
    return model

# === Step 5: Evaluate Model ===
def evaluate_model(model, X_test, y_test_encoded, y_test_raw, label_encoder, sample_size=5):
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_test_classes = np.argmax(y_test_encoded, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

    print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
    print(f"✅ Precision: {precision:.2f}")
    print(f"✅ Recall: {recall:.2f}")
    print(f"✅ F1 Score: {f1:.2f}")

    print("\n🔍 Random Sample Predictions:")
    indices = random.sample(range(len(y_test_raw)), min(sample_size, len(y_test_raw)))
    for i in indices:
         actual = y_test_raw[i]
         predicted = label_encoder.inverse_transform([y_pred_classes[i]])[0]
         print(f"🔹 Actual: {actual} | Predicted: {predicted}")
         speak_prediction_urdu(predicted)

# === Main Execution ===
# Step 1: Load datasets and split into training, validation, and testing
X_train_raw, y_train_raw = load_and_preprocess_data(['Irtaza1.csv', 'Aqsa1.csv', 'Saad1.csv'])
X_val_raw, y_val_raw = load_and_preprocess_data(['Hassan1.csv', 'Numairah1.csv'])  # Use two files for validation
X_test_raw, y_test_raw = load_and_preprocess_data(['Amna1.csv'])

# Optionally split the training data into training and validation sets
import joblib

X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=42)

# Step 2: Scale and Encode
X_train, y_train, scaler, label_encoder, one_hot_encoder = scale_and_encode(X_train_raw, y_train_raw, fit=True)
joblib.dump(label_encoder, 'rnn_label_encoder.joblib')
X_val, y_val, _, _, _ = scale_and_encode(X_val_raw, y_val_raw, fit=False, scaler=scaler, label_encoder=label_encoder, one_hot_encoder=one_hot_encoder)
X_test, y_test_encoded, _, _, _ = scale_and_encode(X_test_raw, y_test_raw, fit=False, scaler=scaler, label_encoder=label_encoder, one_hot_encoder=one_hot_encoder)

# Step 3: Build Model
model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=y_train.shape[1])

# Step 4: Train Model
model = train_model(model, X_train, y_train, X_val, y_val)

# Step 5: Evaluate Model
evaluate_model(model, X_test, y_test_encoded, y_test_raw, label_encoder, sample_size=1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get predictions from the model
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_test_classes = np.argmax(y_test_encoded, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Display confusion matrix with labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

model.save('rnn_model.h5')
import joblib
joblib.dump(scaler, 'rnn_scaler.joblib')