import os

model_path = "C:/Users/rajes/OneDrive/Desktop/realtime_emotion_inference/best_emotion_vgg16.h5"

if os.path.isfile(model_path):
    print("The model file exists.")
else:
    print("The model file does not exist.")
