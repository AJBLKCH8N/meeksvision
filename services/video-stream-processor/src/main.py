import cv2
import face_recognition
import numpy as np
import json
import subprocess
from multiprocessing import Process
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='data/config/config.json'):
    with open(config_path, 'r') as file:
        return json.load(file)

def load_known_faces(encodings_file):
    logging.info(f"Attempting to load encodings from: {encodings_file}")
    try:
        with open(encodings_file, 'r') as file:
            data = json.load(file)
        known_face_encodings = [np.array(encoding) for encoding in data.get('encodings', [])]
        known_face_names = data.get('names', [])
        if not known_face_encodings or not known_face_names:
            logging.warning("Encodings or names are missing from the file.")
        return known_face_encodings, known_face_names
    except FileNotFoundError:
        logging.error(f"Encodings file not found: {encodings_file}")
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from the encodings file.")
    return [], []  # Return empty lists if loading fails

def annotate_frame(frame, face_locations, face_names, known_face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name in known_face_names else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    return frame  # Return the annotated frame


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
    if not video_capture.isOpened():
        logging.error("Failed to open video stream.")
        return
    
    width, height = get_video_dimensions(video_capture)
    logging.info(f"Video stream opened. Dimensions: {width}x{height}")

    command = [
        'ffmpeg',
        '-y',
        '-f', 'image2pipe',
        '-analyzeduration', '1000000',  # Added to increase the duration FFmpeg analyses
        '-probesize', '10000000',  # Increased to allow more data for codec detection
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
        if not ret or frame is None:
            logging.debug("Failed to read frame or frame is empty.")
            break  # Stop processing if no frame is captured

        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB format
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Specify model if using CNN
        logging.debug(f"Detected face locations: {face_locations}")
        
        face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations) if face_locations else []
        face_names = [known_face_names[matches.index(True)] if True in (matches := face_recognition.compare_faces(known_face_encodings, face_encoding)) else "Unknown" for face_encoding in face_encodings]

        annotated_frame = annotate_frame(frame, face_locations, face_names, known_face_names)
        stream_to_ffmpeg(annotated_frame, ffmpeg_process)
        
    video_capture.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    logging.info("Video processing and streaming completed.")

def main():
    config = load_config('data/config/config.json')
    encodings_path = config.get('encodings_file', 'data/faces/known_face_encodings.json')
    known_face_encodings, known_face_names = load_known_faces(encodings_path)
    
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
