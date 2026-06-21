import os

dataset_path = r"D:\B-Tech\S2-Internship\emotion-detection-voice\dataset"

print("Folders:")
print(os.listdir(dataset_path)[:10])

count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            print(file)
            count += 1

            if count == 5:
                break

    if count == 5:
        break