import face_recognition
import numpy as np
import os
import pickle

def encode_known_faces(faces_dir='data/faces', encodings_file='data/faces/known_face_encodings.pkl'):
    known_face_encodings = []
    known_face_names = []

    # Iterate over the folders in faces_dir
    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    filepath = os.path.join(person_dir, filename)
                    image = face_recognition.load_image_file(filepath)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        known_face_encodings.append(encoding[0])
                        known_face_names.append(person_name)

    # Save the encodings and names
    with open(encodings_file, 'wb') as file:
        pickle.dump((known_face_encodings, known_face_names), file)
    print(f"Encoded {len(known_face_encodings)} faces.")

if __name__ == '__main__':
    encode_known_faces()
