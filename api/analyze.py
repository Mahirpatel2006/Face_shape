from http.server import BaseHTTPRequestHandler
import json
import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests

# This is a Vercel Serverless Function which will be deployed as a stand-alone API endpoint.

# IMPORTANT: Model and asset paths are relative to the root of the project,
# not the /api directory. Vercel copies these files to the right place during deployment.
MODEL_PATH = 'Best_RandomForest.pkl'
LANDMARKER_PATH = 'face_landmarker_v2_with_blendshapes.task'

# --- Mediapipe and Model Initialization ---
# This part runs only once when the serverless function is "cold started".
try:
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    with open(MODEL_PATH, 'rb') as f:
        face_shape_model = pickle.load(f)
except Exception as e:
    # If initialization fails, we'll have a global error object
    # to return useful debug information.
    face_landmarker = None
    face_shape_model = None
    INIT_ERROR = e

# --- Helper Functions (from your original app.py) ---
def distance_3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_face_features(coords):
    landmark_indices = {
        'forehead': 10, 'chin': 152, 'left_cheek': 234, 'right_cheek': 454,
        'left_eye': 263, 'right_eye': 33, 'nose_tip': 1
    }
    landmarks_dict = {name: coords[idx] for name, idx in landmark_indices.items()}
    
    features = [
        distance_3d(landmarks_dict['forehead'], landmarks_dict['chin']),
        distance_3d(landmarks_dict['left_cheek'], landmarks_dict['right_cheek']),
        distance_3d(landmarks_dict['left_eye'], landmarks_dict['right_eye']),
        distance_3d(landmarks_dict['nose_tip'], landmarks_dict['left_eye']),
        distance_3d(landmarks_dict['nose_tip'], landmarks_dict['right_eye']),
        distance_3d(landmarks_dict['chin'], landmarks_dict['left_cheek']),
        distance_3d(landmarks_dict['chin'], landmarks_dict['right_cheek']),
        distance_3d(landmarks_dict['forehead'], landmarks_dict['left_eye']),
        distance_3d(landmarks_dict['forehead'], landmarks_dict['right_eye']),
    ]
    return np.array(features)

def get_face_shape_label(label):
    shapes = ["Heart", "Oval", "Round", "Square"]
    return shapes[label]

# --- Main Serverless Handler ---
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Check if initialization failed
        if not face_landmarker or not face_shape_model:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_message = f"Model or landmarker initialization failed: {INIT_ERROR}"
            self.wfile.write(json.dumps({"error": error_message}).encode('utf-8'))
            return

        try:
            # Get the image URL from the request body
            content_len = int(self.headers.get('Content-Length'))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body)
            image_url = data.get('imageUrl')

            if not image_url:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No imageUrl provided"}).encode('utf-8'))
                return

            # Download the image from the URL
            response = requests.get(image_url)
            response.raise_for_status() # Raise an exception for bad status codes
            image_data = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            detection_result = face_landmarker.detect(mp_image)

            if not detection_result.face_landmarks:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No face detected in the image."}).encode('utf-8'))
                return
            
            # Extract features and predict
            landmarks = [[lm.x, lm.y, lm.z] for lm in detection_result.face_landmarks[0]]
            face_features = calculate_face_features(np.array(landmarks))
            face_shape_label = face_shape_model.predict([face_features])[0]
            face_shape = get_face_shape_label(face_shape_label)
            
            # Send successful response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"face_shape": face_shape}).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        return 