import cv2
import face_recognition
import numpy as np
import os
import pickle
import subprocess
from multiprocessing import Process
import json

# Load configurations from a config file within the mounted volume
def load_config(config_path='data/config/config.json'):
    with open(config_path, 'r') as file:
        return json.load(file)

config = load_config()

def load_known_faces():
    encodings_file = config.get('encodings_file', 'known_face_encodings.pkl')
    print(f"Attempting to load encodings from: {encodings_file}")
    with open(encodings_file, 'rb') as file:
        known_face_encodings, known_face_names = pickle.load(file)
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
        print("Frame encoding failed.")

def process_video_stream(stream_url, known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(stream_url)

    width, height = get_video_dimensions(video_capture)

    # Start FFmpeg as a subprocess to send the processed video stream to NGINX
    command = [
        'ffmpeg',
        '-y',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-s', f'{width}x{height}',  # Use the source's dimensions
        '-i', '-',
        '-vcodec', 'libx264',
        '-preset', 'veryfast',
        '-f', 'flv',
        'rtmp://streaming-service/live/stream'  # Stream to the NGINX RTMP application
    ]
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"Face locations: {face_locations}")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
        annotate_frame(frame, face_locations, face_names, known_face_names)

        # Stream the processed frame to FFmpeg
        stream_to_ffmpeg(frame, ffmpeg_process)

        # cv2.imshow('Video', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    video_capture.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

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
