import cv2

# Initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    """
    Detect faces in a video frame.

    Parameters:
    - frame: The video frame, as a numpy array.

    Returns:
    - An array of rectangles where faces were detected. Each rectangle is (x, y, w, h).
    """
    # Convert the frame to grayscale as the Haar Cascade model expects grayscale inputs
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces

if __name__ == "__main__":
    # For testing: Load an image and try to detect faces
    test_img = cv2.imread('test.jpg')
    detected_faces = detect_faces(test_img)

    # Draw rectangles around detected faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Save the result
    cv2.imwrite('detected_faces.jpg', test_img)
