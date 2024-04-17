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
            data = json.load(file)
        known_encodings = {name: [np.array(enc, dtype='float64') for enc in encodings] for name, encodings in data.items()}
        if not known_encodings:
            logging.error("Encodings are empty after loading.")
        else:
            logging.info(f"Loaded encodings for {len(known_encodings)} people.")
        return known_encodings
    except FileNotFoundError:
        logging.error(f"The file {path} was not found.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the file {path}.")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
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
        current_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = []
        for current_encoding in current_encodings:
            if known_encodings:
                distances = {name: np.linalg.norm(encodings - current_encoding, axis=1) for name, encodings in known_encodings.items() if encodings}
                if not distances:  # Check if distances dictionary is empty
                    logging.error("Distances computation returned empty. No known encodings to compare.")
                    face_names.append('Unknown')
                    logging.info("Recognized face: Unknown")
                    continue
                min_distance = {name: min(dist) if dist else float('inf') for name, dist in distances.items()}
                best_match = min(min_distance, key=min_distance.get)
                if min_distance[best_match] < 0.6:
                    face_names.append(best_match)
                    logging.info(f"Recognized face: {best_match}")
                else:
                    face_names.append('Unknown')
                    logging.info("Recognized face: Unknown")
            else:
                logging.error("No known encodings loaded.")
                face_names.append('Unknown')
                logging.info("Recognized face: Unknown")
        
        return face_locations, face_names
    except Exception as e:
        logging.error(f"Failed to recognize faces: {e}, frame shape: {frame.shape}")
        return face_locations, ['Unknown'] * len(face_locations)


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
