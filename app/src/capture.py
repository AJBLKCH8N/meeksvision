""" Video Stream Capture Module """

import cv2

def capture_stream(stream_url):
    cap = cv2.VideoCapture(stream_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame, exiting...")
            break
        yield frame
        
if __name__ == "__main__":
    stream_url = "rtsp://<CAMERA STREAM>"
    for frame in capture_stream(stream_url):
        cv2.imshow('Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()