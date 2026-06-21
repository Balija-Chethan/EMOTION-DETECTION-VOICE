import librosa
import numpy as np
import joblib
from pydub import AudioSegment
import os

# Load trained model
model = joblib.load("emotion_model.pkl")


def convert_mp3_to_wav(file_path):

    if file_path.lower().endswith(".mp3"):

        wav_path = file_path.replace(".mp3", ".wav")

        audio = AudioSegment.from_mp3(file_path)

        audio.export(
            wav_path,
            format="wav"
        )

        return wav_path

    return file_path


def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=22050,
        mono=True,
        duration=3,
        offset=0.5
    )

    audio = librosa.util.normalize(audio)

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )
    return np.mean(mfccs.T, axis=0)


def predict_emotion(file_path):

    file_path = convert_mp3_to_wav(file_path)

    features = extract_features(file_path)

    prediction = model.predict([features])[0]

    confidence = max(
        model.predict_proba([features])[0]
    ) * 100

    return prediction, round(confidence, 2)
