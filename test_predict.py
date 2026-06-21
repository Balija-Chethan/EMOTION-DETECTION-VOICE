from predict import predict_emotion

audio_file = r"D:\B-Tech\S2-Internship\emotion-detection-voice\uploads\Singari.mp3"

emotion, confidence = predict_emotion(audio_file)

print("Emotion:", emotion)
print("Confidence:", confidence, "%")