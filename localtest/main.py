import cv2
import face_recognition
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='../data/config/config.json'):
    with open(config_path, 'r') as file:
        return json.load(file)

def load_known_faces(encodings_file):
    logging.info(f"Attempting to load encodings from: {encodings_file}")
    try:
        with open(encodings_file, 'r') as file:
            data = json.load(file)
        known_face_encodings = [np.array(encoding) for encoding in data['encodings']]
        known_face_names = data['names']
        if not known_face_encodings or not known_face_names:
            logging.warning("Encodings or names are missing from the file.")
        return known_face_encodings, known_face_names
    except FileNotFoundError:
        logging.error(f"Encodings file not found: {encodings_file}")
        return [], []
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from the encodings file.")
        return [], []

def annotate_frame(frame, face_locations, face_names, known_face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name in known_face_names else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

def process_video_stream(stream_url, known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(stream_url)
    if not video_capture.isOpened():
        logging.error("Failed to open video stream.")
        return

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (frame_width, frame_height))

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logging.debug("Failed to read frame.")
                break

            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations) if face_locations else []
            face_names = [known_face_names[matches.index(True)] if True in (matches := face_recognition.compare_faces(known_face_encodings, face_encoding)) else "Unknown" for face_encoding in face_encodings]

            annotate_frame(frame, face_locations, face_names, known_face_names)
            video_writer.write(frame)  # Save processed frame

    finally:
        video_capture.release()
        video_writer.release()
        logging.info("Video processing completed and resources released.")

def main():
    config = load_config()
    encodings_path = config.get('encodings_file', '../data/encoding/known_face_encodings.json')
    known_face_encodings, known_face_names = load_known_faces(encodings_path)
    stream_url = config.get('camera_url', 'rtsp://tapo-dev:@!33Cisco33!@@192.168.1.16:554/stream1')
    process_video_stream(stream_url, known_face_encodings, known_face_names)

if __name__ == '__main__':
    main()