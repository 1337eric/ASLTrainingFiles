import sys
import json
import cv2
import numpy as np
import base64
import traceback
import tensorflow as tf
import mediapipe as mp
import threading
import os

tf.keras.utils.disable_interactive_logging()
tf.config.list_physical_devices('GPU')


def load_model_and_labels(model_path, label_encoder_path):
    model = tf.keras.models.load_model(model_path)
    label_encoder_classes = np.load(label_encoder_path, allow_pickle=True)
    return model, label_encoder_classes

def predict_gesture(model, label_encoder_classes, frames_data):
    # Reshape and flatten the input data
    frames_data = frames_data.reshape(30, -1)  # Flatten each frame
    frames_data = frames_data.astype('float32')
    
    # Normalize the data
    max_val = np.max(np.abs(frames_data))
    if max_val > 0:
        frames_data = frames_data / max_val
    
    frames_data = np.expand_dims(frames_data, axis=0)
    
    # Make prediction
    prediction = model.predict(frames_data)
    predicted_class = label_encoder_classes[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return predicted_class, confidence

global double_frame_buffer
double_frame_buffer = []

def double_buffer():
    global double_frame_buffer
    for line in sys.stdin:
        double_frame_buffer.append(line)
        if len(double_frame_buffer) > 60:
            double_frame_buffer.pop(0)

double_buffer_thread = threading.Thread(target=double_buffer)
double_buffer_thread.start()

previous_frame = None

def process_frame_continuous(model, label_encoder_classes):
    global double_frame_buffer
    global previous_frame

    delta_frames = 0

    whole_sentence = ""
    last_gesture = None
    frozen_counter = 0
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    frames = []
    # Updated key points to include face tracking points
    KEY_FACE_POINTS = [33, 133, 362, 263, 61, 291]
    
    # Arm tracking points
    KEY_ARM_POINTS = [
        11,  # Left shoulder
        13,  # Left elbow
        15,  # Left wrist
        12,  # Right shoulder
        14,  # Right elbow
        16   # Right wrist
    ]
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while sys.stdin:
            try:
                line = double_frame_buffer[-1]
                if not line.strip():  # Skip empty lines
                    continue
            except IndexError:
                continue
            try:
                data = json.loads(line)
                frame_b64 = data.get('frame', '').replace("data:image/webp;base64,","")
                
                if not frame_b64:
                    print("Warning: Empty frame data received")
                    continue
                
                # Decode base64 to image
                img_bytes = base64.b64decode(frame_b64 + '=' * (-len(frame_b64) % 4))
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                # Process frame with MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks with updated face and arm points
                frame_landmarks = np.zeros((6 + 6 + 21 * 2, 3))  # 6 face + 6 arm + 21 per hand
                
                # Face landmarks (6 key points)
                if results.face_landmarks:
                    for idx, point_idx in enumerate(KEY_FACE_POINTS):
                        lm = results.face_landmarks.landmark[point_idx]
                        frame_landmarks[idx] = [lm.x, lm.y, lm.z]
                
                # Arm landmarks (6 key points)
                if results.pose_landmarks:
                    start_idx = 6  # After face points
                    for idx, point_idx in enumerate(KEY_ARM_POINTS):
                        lm = results.pose_landmarks.landmark[point_idx]
                        frame_landmarks[start_idx + idx] = [lm.x, lm.y, lm.z]
                
                # Right hand landmarks
                if results.right_hand_landmarks:
                    start_idx = 6 + 6  # After face and arm points
                    for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                        frame_landmarks[start_idx + idx] = [lm.x, lm.y, lm.z]
                
                # Left hand landmarks
                if results.left_hand_landmarks:
                    start_idx = 6 + 6 + 21  # After face, arm, and right hand
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

                # Draw arm landmarks and connect them with lines
                if results.pose_landmarks:
                    for i in range(0, len(KEY_ARM_POINTS), 3):
                        shoulder = results.pose_landmarks.landmark[KEY_ARM_POINTS[i]]
                        elbow = results.pose_landmarks.landmark[KEY_ARM_POINTS[i+1]]
                        wrist = results.pose_landmarks.landmark[KEY_ARM_POINTS[i+2]]
                        
                        # Convert to pixel coordinates
                        shoulder_pos = (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0]))
                        elbow_pos = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
                        wrist_pos = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
                        
                        # Draw lines
                        cv2.line(image, shoulder_pos, elbow_pos, (255, 0, 0), 2)
                        cv2.line(image, elbow_pos, wrist_pos, (255, 0, 0), 2)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                if len(frames) == 30:
                    frames_array = np.array(frames)
                    gesture, confidence = predict_gesture(model, label_encoder_classes, frames_array)
                    if gesture != last_gesture and delta_frames >= 30 and confidence > 0.8 and gesture != "nothing":
                        delta_frames = 0
                        whole_sentence += gesture + " "
                        last_gesture = gesture
                        #print(whole_sentence)
                    else:
                        delta_frames += 1
                    cv2.putText(image, f'Gesture: {gesture} ({confidence:.2f})', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if len(frames) > 29:
                    frames.pop(0)

                cv2.imshow('Continuous Gesture Recognition', image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if previous_frame == frame_b64:
                    frozen_counter += 1
                else:
                    frozen_counter = 0

                if frozen_counter == 10:
                    print(whole_sentence)
                if frozen_counter > 10:
                    whole_sentence = ""

                previous_frame = frame_b64

                sys.stdout.flush()
                
            except Exception as e:
                print(f"Error processing input: {e}")
                traceback.print_exc()
    
    cv2.destroyAllWindows()

def main():
    model_path = "../ASL/gesture_model.h5"
    label_encoder_path = "../ASL/label_encoder_classes.npy"
    
    model, label_encoder_classes = load_model_and_labels(model_path, label_encoder_path)
    process_frame_continuous(model, label_encoder_classes)

if __name__ == "__main__":
    main()