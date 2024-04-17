import face_recognition
import cv2
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_known_encodings(path):
    try:
        with open(path, 'r') as file:
            known_encodings = json.load(file)
        logging.debug(f"Raw encodings loaded: {known_encodings}")  # Detailed debug log
        known_encodings = {name: [np.array(encoding) for encoding in encodings] for name, encodings in known_encodings.items()}
        logging.info("Known encodings processed and loaded successfully from {}".format(path))
        return known_encodings
    except FileNotFoundError:
        logging.error("The file {} was not found.".format(path))
        return {}
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from the file {}.".format(path))
        return {}


known_encodings = load_known_encodings('/app/encoding/known_face_encodings.json')

def detect_faces(frame):
    try:
        logging.debug(f"Processing frame for face detection: shape={frame.shape}")
        face_locations = face_recognition.face_locations(frame)
        logging.info("Detected {} faces".format(len(face_locations)))
        return face_locations
    except Exception as e:
        logging.error(f"Failed to detect faces: {e}, frame shape: {frame.shape}")
        return []

def recognize_faces(frame, face_locations):
    try:
        logging.debug(f"Starting face recognition on frame with shape={frame.shape} and locations={face_locations}")
        current_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = []
        for current_encoding in current_encodings:
            matches = {name: face_recognition.compare_faces(encodings, current_encoding, tolerance=0.6) for name, encodings in known_encodings.items()}
            face_distances = {name: face_recognition.face_distance(encodings, current_encoding) for name, encodings in known_encodings.items()}
            best_match = min(face_distances, key=face_distances.get, default="Unknown")
            name = best_match if face_distances[best_match] < 0.6 else "Unknown"
            face_names.append(name)
        
        logging.info("Faces recognized: {}".format(", ".join(face_names) if face_names else "No faces recognized"))
        return face_locations, face_names
    except Exception as e:
        logging.error(f"Failed to recognize faces: {e}, frame shape: {frame.shape}")
        return face_locations, []


def annotate_frame(frame, face_locations, face_names):
    try:
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            logging.debug(f"Annotating face: {name}, location: {(left, top, right, bottom)}")
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        logging.info("Frame annotated")
        return frame
    except Exception as e:
        logging.error(f"Failed to annotate frame: {e}, face details: {zip(face_locations, face_names)}")
        return frame

