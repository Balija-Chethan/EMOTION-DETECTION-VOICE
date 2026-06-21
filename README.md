# 🎤 Human Emotion Detection from Voice

A Machine Learning-based web application that detects human emotions from voice recordings. Users can upload WAV or MP3 audio files, and the system predicts the speaker's emotion along with a confidence score.

## 🚀 Live Demo

🔗 Live Application: https://emotion-detection-voice-i8c8yaun7gbjv2mgqcu9vt.streamlit.app

## 📌 Features

* Upload WAV and MP3 audio files
* Automatic MP3 to WAV conversion
* Real-time emotion prediction
* Confidence score display
* Audio playback support
* Interactive Streamlit interface
* Cloud deployment using Streamlit Community Cloud

## 🎯 Supported Emotions

* 😊 Happy
* 😢 Sad
* 😡 Angry
* 😨 Fearful
* 😌 Calm
* 😐 Neutral
* 😲 Surprised
* 🤢 Disgust

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Classifier

### Audio Processing

* Librosa
* Pydub
* FFmpeg
* SoundFile

### Web Framework

* Streamlit

### Dataset

* RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

## 🧠 Model Workflow

1. Audio file upload
2. MP3 to WAV conversion (if required)
3. Audio preprocessing and normalization
4. MFCC feature extraction
5. Emotion prediction using trained Random Forest model
6. Display emotion and confidence score

## 📊 Model Performance

* Dataset: RAVDESS Emotional Speech Dataset
* Total Samples: 2880
* Algorithm: Random Forest Classifier
* Accuracy: 92.71%

## 📂 Project Structure

emotion-detection-voice/

├── streamlit_app.py

├── predict.py

├── train.py

├── emotion_model.pkl

├── requirements.txt

├── packages.txt

├── templates/

├── uploads/

└── README.md

## ⚙️ Installation

Clone the repository:

git clone https://github.com/Balija-Chethan/emotion-detection-voice.git

Navigate to the project directory:

cd emotion-detection-voice

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run streamlit_app.py

## 📷 Application Preview

Upload a WAV or MP3 file and the system will:

* Play the uploaded audio
* Predict the emotion
* Display confidence score
* Show emotion emoji

## 🌟 Future Enhancements

* Real-time microphone recording
* Deep Learning (CNN/LSTM) model integration
* Emotion trend analysis
* Multi-language emotion recognition
* Improved UI and analytics dashboard

## 👨‍💻 Author

Balija Chethan

GitHub:
https://github.com/Balija-Chethan

## 📜 License

This project is developed for educational and portfolio purposes.
