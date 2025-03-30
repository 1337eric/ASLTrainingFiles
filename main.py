import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic # the model
mp_drawing = mp.solutions.drawing_utils # drawing utility
mp_face_mesh = mp.solutions.face_mesh # for face connections

def detect(image, model):
	# Calculate dimensions
	image_height, image_width, _ = image.shape
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert color
	image.flags.writeable = False 
	results = model.process(image) # process the image
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert back
	return image, results

def draw(image, results):
	# Draw face landmarks
	if results.face_landmarks:
		mp_drawing.draw_landmarks(
			image,
			results.face_landmarks,
			mp_face_mesh.FACEMESH_CONTOURS,
			mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
			mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
		)
	# Draw pose landmarks
	if results.pose_landmarks:
		mp_drawing.draw_landmarks(
			image,
			results.pose_landmarks,
			mp_holistic.POSE_CONNECTIONS,
			mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
			mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
		)
	# Draw left hand landmarks
	if results.left_hand_landmarks:
		mp_drawing.draw_landmarks(
			image,
			results.left_hand_landmarks,
			mp_holistic.HAND_CONNECTIONS,
			mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
			mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
		)
	# Draw right hand landmarks
	if results.right_hand_landmarks:
		mp_drawing.draw_landmarks(
			image,
			results.right_hand_landmarks,
			mp_holistic.HAND_CONNECTIONS,
			mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
			mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
		)

def extract(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

cap = cv2.VideoCapture(0)
# Set specific dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
	while cap.isOpened():
		ret, frame = cap.read() # get camera
		if not ret:
			print("Failed to grab frame")
			break

		image, results = detect(frame, holistic) # do the detection
		draw(image, results) # draw the landmarks

		cv2.imshow("Holistic Model Detection", image) # show the processed image
		if cv2.waitKey(10) & 0xFF == ord("q"): # if q clicked then break
			break

	cap.release()
	cv2.destroyAllWindows()