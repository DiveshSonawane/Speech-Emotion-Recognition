import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import tempfile
import os
import soundfile as sf
import io
import keras

# ---------------- LOAD SAVED OBJECTS ----------------
scaler = joblib.load("C:/Users/DELL/models/scaler.pkl")
encoder = joblib.load("C:/Users/DELL/models/label_encoder.pkl")
xgb_model = joblib.load("C:/Users/DELL/models/xgb_model.pkl")
mlp_model = tf.keras.models.load_model("C:/Users/DELL/models/mlp_model.keras")


# ---------------- FEATURE EXTRACTION ----------------
def extract_features(audio_input):
    if audio_input is None:
        return None

    audio_bytes = audio_input.read()
    audio_buffer = io.BytesIO(audio_bytes)

    y, sr = librosa.load(audio_buffer, duration=3, offset=0.5)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(
        librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T,
        axis=0
    )

    return np.hstack([mfcc, chroma, mel, contrast, tonnetz])



# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("🎤 Speech Emotion Recognition")
st.write("Upload a WAV file and predict emotion using MLP & XGBoost")

uploaded_file = st.file_uploader("Upload a .wav file", type="wav")

if uploaded_file is not None:
    features = extract_features(uploaded_file)
    features = scaler.transform([features])

    # MLP
    features_mlp = features.reshape(1, -1)
    pred_mlp = mlp_model.predict(features_mlp)
    emotion_mlp = encoder.inverse_transform([np.argmax(pred_mlp)])[0]

    # XGBoost
    pred_xgb = xgb_model.predict(features)[0]
    emotion_xgb = encoder.inverse_transform([pred_xgb])[0]

    st.success("Prediction complete!")
    st.write(f"🎧 **MLP Prediction:** {emotion_mlp}")
    st.write(f"🌲 **XGBoost Prediction:** {emotion_xgb}")
else:
    st.info("Please upload a WAV audio file to begin.")
