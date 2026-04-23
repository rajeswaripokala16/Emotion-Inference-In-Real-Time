import cv2
import numpy as np
import json
import argparse
import os
from tensorflow.keras.models import load_model

# --- Config ---
MODEL_PATH = "best_emotion_vgg16.h5"
LABELS_JSON = "labels.json"
IMG_SIZE = (224, 224)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SCALE = 1.1
MIN_NEIGHBORS = 5
# ----------------

def load_labels(labels_json=LABELS_JSON):
    if not os.path.isfile(labels_json):
        print(f"Labels file not found: {labels_json}")
        return {}
    with open(labels_json, "r") as f:
        labels_map = json.load(f)
    labels_map = {int(k): v for k, v in labels_map.items()}
    return labels_map

def load_emotion_model(model_path=MODEL_PATH):
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def preprocess_face(face):
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, IMG_SIZE)
        arr = face.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        print(f"Error in preprocessing face: {e}")
        return None

def infer_emotion(model, inp):
    try:
        preds = model.predict(inp)
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx])
        return class_idx, confidence
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None, None

def main(model_path=MODEL_PATH, labels_json=LABELS_JSON):
    model = load_emotion_model(model_path)
    if model is None:
        return

    labels = load_labels(labels_json)
    if not labels:
        print("Could not load labels. Exiting.")
        return

    print("Loaded model and labels:", labels)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=SCALE, minNeighbors=MIN_NEIGHBORS)

        for (x, y, w, h) in faces:
            pad = int(0.15 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            inp = preprocess_face(face_crop)
            if inp is None:
                continue

            class_idx, confidence = infer_emotion(model, inp)
            if class_idx is None:
                continue

            label = labels.get(class_idx, "Unknown")
            text = f"{label}: {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Emotion Detection (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
