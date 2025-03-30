import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import os

def load_model_and_labels(model_path, label_encoder_path):
    model = tf.keras.models.load_model(model_path)
    label_encoder_classes = np.load(label_encoder_path, allow_pickle=True)
    return model, label_encoder_classes

def collect_frames_for_prediction():
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    frames = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while len(frames) < 30:
            ret, frame = cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks with updated face points
            frame_landmarks = np.zeros((48, 3))  # 6 face + 42 hand points
            
            # Face landmarks (6 key points)
            KEY_FACE_POINTS = [
                33,  # Left eye left corner
                133, # Right eye right corner
                362, # Right eye left corner
                263, # Left eye right corner
                61,  # Mouth left corner
                291  # Mouth right corner
            ]
            
            if results.face_landmarks:
                for idx, point_idx in enumerate(KEY_FACE_POINTS):
                    lm = results.face_landmarks.landmark[point_idx]
                    frame_landmarks[idx] = [lm.x, lm.y, lm.z]
            
            # Right hand landmarks
            if results.right_hand_landmarks:
                start_idx = 6  # After face points
                for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                    frame_landmarks[start_idx + idx] = [lm.x, lm.y, lm.z]
            
            # Left hand landmarks
            if results.left_hand_landmarks:
                start_idx = 6 + 21  # After face and right hand
                for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                    frame_landmarks[start_idx + idx] = [lm.x, lm.y, lm.z]

            frames.append(frame_landmarks)

            # Draw landmarks
            if results.face_landmarks:
                for point_idx in KEY_FACE_POINTS:
                    pos = results.face_landmarks.landmark[point_idx]
                    x = int(pos.x * image.shape[1])
                    y = int(pos.y * image.shape[0])
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.putText(image, f'Collecting frame: {len(frames)}/30', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Prediction Collection', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames)

def predict_gesture(model, label_encoder_classes, frames_data):
    # Reshape and flatten the input data
    frames_data = frames_data.reshape(30, -1)  # Flatten each frame to 144 values
    frames_data = frames_data.astype('float32')
    frames_data = frames_data / np.max(frames_data)
    frames_data = np.expand_dims(frames_data, axis=0)
    
    # Make prediction
    prediction = model.predict(frames_data)
    predicted_class = label_encoder_classes[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return predicted_class, confidence

def collect_and_predict_continuous(model, label_encoder_classes):
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    frames = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks with updated face points
            frame_landmarks = np.zeros((48, 3))  # 6 face + 42 hand points
            
            # Face landmarks (6 key points)
            KEY_FACE_POINTS = [
                33,  # Left eye left corner
                133, # Right eye right corner
                362, # Right eye left corner
                263, # Left eye right corner
                61,  # Mouth left corner
                291  # Mouth right corner
            ]
            
            if results.face_landmarks:
                for idx, point_idx in enumerate(KEY_FACE_POINTS):
                    lm = results.face_landmarks.landmark[point_idx]
                    frame_landmarks[idx] = [lm.x, lm.y, lm.z]
            
            # Right hand landmarks
            if results.right_hand_landmarks:
                start_idx = 6  # After face points
                for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                    frame_landmarks[start_idx + idx] = [lm.x, lm.y, lm.z]
            
            # Left hand landmarks
            if results.left_hand_landmarks:
                start_idx = 6 + 21  # After face and right hand
                for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                    frame_landmarks[start_idx + idx] = [lm.x, lm.y, lm.z]

            frames.append(frame_landmarks)
            
            # Keep only the last 30 frames
            if len(frames) > 30:
                frames.pop(0)

            # Draw landmarks
            if results.face_landmarks:
                for point_idx in KEY_FACE_POINTS:
                    pos = results.face_landmarks.landmark[point_idx]
                    x = int(pos.x * image.shape[1])
                    y = int(pos.y * image.shape[0])
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Make prediction if we have enough frames
            if len(frames) == 30:
                frames_array = np.array(frames)
                gesture, confidence = predict_gesture(model, label_encoder_classes, frames_array)
                cv2.putText(image, f'Gesture: {gesture} ({confidence:.2f})', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Continuous Gesture Recognition', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def main():
    model_path = "gesture_model.h5"
    label_encoder_path = "label_encoder_classes.npy"
    
    model, label_encoder_classes = load_model_and_labels(model_path, label_encoder_path)
    collect_and_predict_continuous(model, label_encoder_classes)

if __name__ == "__main__":
    main()