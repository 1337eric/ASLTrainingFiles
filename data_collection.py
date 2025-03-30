import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def initialize_capture():
    cap = cv2.VideoCapture(0)
    return cap

def create_gesture_directory(gesture_name):
    base_path = os.path.join('gestures', gesture_name)
    os.makedirs(base_path, exist_ok=True)
    return base_path

# Remove this function as it's no longer needed
# def save_recording_data(frame_data, gesture_path, recording_num):
#     recording_dir = os.path.join(gesture_path, str(recording_num))
#     os.makedirs(recording_dir, exist_ok=True)
#     np.save(os.path.join(recording_dir, f'{recording_num}.npy'), np.array(frame_data))

def save_frame_data(frame_landmarks, gesture_path, recording_num, frame_num):
    recording_dir = os.path.join(gesture_path, str(recording_num))
    os.makedirs(recording_dir, exist_ok=True)
    np.save(os.path.join(recording_dir, f'{frame_num}.npy'), frame_landmarks)

def collect_frames(cap, gesture_path, recording_num, frames_to_collect=30):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frames_collected = 0
        
        # Updated key points to include arm tracking points
        KEY_FACE_POINTS = [
            33,  # Left eye left corner
            133, # Right eye right corner
            362, # Right eye left corner
            263, # Left eye right corner
            61,  # Mouth left corner
            291  # Mouth right corner
        ]

        # Arm tracking points
        KEY_ARM_POINTS = [
            11,  # Left shoulder
            13,  # Left elbow
            15,  # Left wrist
            12,  # Right shoulder
            14,  # Right elbow
            16   # Right wrist
        ]
        
        # Remove the countdown from here since we'll do it in the main function
        
        while frames_collected < frames_to_collect:
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detections
            results = holistic.process(image)
            
            # Convert back to BGR for rendering
            image.flags.writeable = True
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

            # Save each frame individually
            save_frame_data(frame_landmarks, gesture_path, recording_num, frames_collected)
            frames_collected += 1

            # Draw landmarks
            # Only draw the 4 key face points instead of all face landmarks
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

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Display frame count
            cv2.putText(image, f'Collecting frame: {frames_collected}/{frames_to_collect}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Remove this line as we're already saving frames individually
        # save_recording_data(frame_data, gesture_path, recording_num)

def main():
    os.makedirs('gestures', exist_ok=True)
    
    gesture_name = input("Enter the name of the gesture to record: ").strip()
    num_recordings = int(input("How many times would you like to record this gesture? "))
    
    gesture_path = create_gesture_directory(gesture_name)
    cap = initialize_capture()
    
    recording_count = 0
    start_recording = False
    while recording_count < num_recordings:
        ret, frame = cap.read()
        if not ret:
            continue
        
        if not start_recording:
            cv2.putText(frame, f'Recording {recording_count + 1}/{num_recordings}. Press "s" to start', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                start_recording = True
        else:
            # Improved countdown that doesn't freeze
            countdown_start = time.time()
            countdown_seconds = 3
            
            while time.time() - countdown_start < countdown_seconds:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Calculate remaining time
                remaining = countdown_seconds - int(time.time() - countdown_start)
                
                # Only show countdown if it's 3, 2, or 1
                if 1 <= remaining <= 3:
                    cv2.putText(frame, f'Starting in {remaining}...', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Data Collection', frame)
                
                # Short wait to keep UI responsive
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            collect_frames(cap, gesture_path, recording_count)
            print(f"Completed recording {recording_count + 1}/{num_recordings} for gesture '{gesture_name}'")
            recording_count += 1

            # Don't reset start_recording, so it continues automatically
            # Instead, add a brief pause between recordings
            if recording_count < num_recordings:
                pause_start = time.time()
                while time.time() - pause_start < 1:  # 1 second pause
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(frame, f'Preparing for next recording ({recording_count + 1}/{num_recordings})...', 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Data Collection', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()