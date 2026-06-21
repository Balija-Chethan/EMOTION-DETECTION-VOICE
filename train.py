import os
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATASET_PATH = r"D:\B-Tech\S2-Internship\emotion-detection-voice\dataset"


emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def extract_features(file_path):
    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    return np.mean(mfccs.T, axis=0)


X = []
y = []

print("Loading dataset...")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            emotion_code = file.split("-")[2]
            emotion = emotion_dict[emotion_code]

            file_path = os.path.join(root, file)

            try:
                features = extract_features(file_path)

                X.append(features)
                y.append(emotion)

            except Exception as e:
                print("Error:", file_path)


print("Total samples:", len(X))

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", round(accuracy * 100, 2), "%")

joblib.dump(
    model,
    "emotion_model.pkl"
)

print("Model saved as emotion_model.pkl")