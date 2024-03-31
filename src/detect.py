""" Face Detection Capture Module """

import cv2

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

if __name__ == "__main__":
    # This is for testing; integrate with capture.py in the actual project
    frame = cv2.imread('test.jpg')  # Replace with a path to a test image
    detected_frame = detect_faces(frame)
    cv2.imshow('Detected Faces', detected_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
