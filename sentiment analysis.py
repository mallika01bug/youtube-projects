import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained facial expression model
model = load_model("fer2013_mini_XCEPTION.119-0.65.hdf5",
                   compile=False)
# Updated model path

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Emotion labels (Ensure these match the model)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy',
                  'Neutral', 'Sad', 'Surprise']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe FaceMesh
    result = face_mesh.process(rgb_frame)
    
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Extract bounding box
            x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)
            
            # Extract face ROI
            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size == 0:
                continue
            
            # Preprocess face ROI for model input
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi = np.expand_dims(face_roi, axis=[0, -1]) / 255.0
            
            # Predict emotion
            predictions = model.predict(face_roi)
            emotion = emotion_labels[np.argmax(predictions)]
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow("Real-Time Sentiment Analysis", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
