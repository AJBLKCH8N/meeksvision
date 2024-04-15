import cv2
import face_recognition
import numpy as np
import json
import subprocess
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

def stream_to_ffmpeg(frame, ffmpeg_process):
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        ffmpeg_process.stdin.write(buffer.tobytes())
    else:
        logging.error("Frame encoding failed.")

def process_video_stream(stream_url, known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(stream_url)
    if not video_capture.isOpened():
        logging.error("Failed to open video stream.")
        return

    width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    command = [
        'ffmpeg',
        '-y',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-s', f'{width}x{height}',
        '-i', '-',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'veryfast',
        '-f', 'flv',
        'rtmp://192.168.1.5/live/stream'
    ]
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logging.debug("Failed to read frame.")
                break

            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            if face_locations:
                face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
                if face_landmarks:  # Check if landmarks are valid before processing
                    try:
                        face_encodings = [face_recognition.face_encodings(rgb_frame, [landmark])[0] for landmark in face_landmarks]
                        face_names = [
                            known_face_names[matches.index(True)] if True in (matches := face_recognition.compare_faces(known_face_encodings, face_encoding)) else "Unknown"
                            for face_encoding in face_encodings
                        ]
                        annotate_frame(frame, face_locations, face_names, known_face_names)
                        stream_to_ffmpeg(frame, ffmpeg_process)  # Stream processed frame
                    except Exception as e:
                        logging.error(f"Error processing face data: {str(e)}")
                else:
                    logging.info("No valid landmarks detected.")
            else:
                logging.debug("No faces detected.")

            # Monitor FFmpeg's output
            while True:
                output = ffmpeg_process.stderr.readline()
                if output == '' and ffmpeg_process.poll() is not None:
                    break
                if output:
                    logging.debug(output.strip())

    finally:
        video_capture.release()
        ffmpeg_process.stdin.close()
        try:
            ffmpeg_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
        logging.info("Video processing and streaming completed.")

def main():
    config = load_config()
    encodings_path = config.get('encodings_file', 'data/encoding/known_face_encodings.json')
    known_face_encodings, known_face_names = load_known_faces(encodings_path)
    stream_url = config.get('camera_url', 'rtsp://tapo-dev:@!33Cisco33!@@192.168.1.16:554/stream1')
    process_video_stream(stream_url, known_face_encodings, known_face_names)

if __name__ == '__main__':
    main()
