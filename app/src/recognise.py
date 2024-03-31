""" Face Recognition Capture Module """

import face_recognition
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, known_faces_dir='data/faces'):
        self.known_face_encodings = []
        self.known_face_names = []

        # Load and encode faces from the known faces directory
        for name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, name)

            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                image = face_recognition.load_image_file(image_path)
                # Assuming each image contains exactly one face
                encoding = face_recognition.face_encodings(image)[0]

                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)

    def recognize_faces(self, frame):
        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        return face_locations, face_names

if __name__ == "__main__":
    # Example usage
    recognizer = FaceRecognizer()

    # Load a test image and recognize faces
    test_image = face_recognition.load_image_file('test.jpg')
    face_locations, face_names = recognizer.recognize_faces(test_image)

    print("Found faces:", face_names)
