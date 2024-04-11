import cv2
import face_recognition
import numpy as np
import os
import json
import subprocess
from multiprocessing import Process
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configurations from a config file within the mounted volume
def load_config(config_path='data/config/config.json'):
    with open(config_path, 'r') as file:
        return json.load(file)

config = load_config()

def load_known_faces():
    encodings_file = config.get('encodings_file', 'data/faces/known_face_encodings.json')
    logging.info(f"Attempting to load encodings from: {encodings_file}")
    with open(encodings_file, 'r') as file:
        data = json.load(file)
        known_face_encodings = [np.array(encoding) for encoding in data['encodings']]
        known_face_names = data['names']
    return known_face_encodings, known_face_names

def annotate_frame(frame, face_locations, face_names, known_face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name in known_face_names else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

def get_video_dimensions(video_capture):
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

def stream_to_ffmpeg(frame, ffmpeg_process):
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        ffmpeg_process.stdin.write(buffer.tobytes())
    else:
        logging.error("Frame encoding failed.")

def process_video_stream(stream_url, known_face_encodings, known_face_names, max_retries=3, retry_delay=1):
    video_capture = cv2.VideoCapture(stream_url)

    width, height = get_video_dimensions(video_capture)
    logging.info(f"Video stream opened. Dimensions: {width}x{height}")

    command = [
        'ffmpeg',
        '-y',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-s', f'{width}x{height}',
        '-i', '-',
        '-vcodec', 'libx264',
        '-preset', 'veryfast',
        '-f', 'flv',
        'rtmp://streaming-service/live/stream'
    ]
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)
    logging.info("FFmpeg process started.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        # Try to detect face locations in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        logging.debug(f"Detected face locations: {face_locations}")

        # Attempt to encode faces based on detected locations
        try:
            if face_locations:  # Ensure there are detected faces before encoding
                face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
                # Process face encodings as before
            else:
                logging.debug("No faces detected in this frame.")
        except Exception as e:
            logging.error(f"Error during face encoding: {e}")

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

        annotate_frame(frame, face_locations, face_names, known_face_names)
        stream_to_ffmpeg(frame, ffmpeg_process)

    video_capture.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    logging.info("Video processing and streaming completed.")

def main():
    known_face_encodings, known_face_names = load_known_faces()
    stream_urls = config.get('camera_urls', ['rtsp://localhost/stream1'])
    processes = []
    for url in stream_urls:
        process = Process(target=process_video_stream, args=(url, known_face_encodings, known_face_names))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

if __name__ == '__main__':
    main()
