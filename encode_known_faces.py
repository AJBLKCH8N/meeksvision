import face_recognition
import numpy as np
import os
import json

def encode_known_faces(faces_dir='data/faces', encodings_file='data/encoding/known_face_encodings.json'):
    known_faces = {}  # This will store the encodings indexed by names

    # Iterate over the folders in faces_dir
    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if os.path.isdir(person_dir):
            # List to hold all encodings for this person
            person_encodings = []
            for filename in os.listdir(person_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    filepath = os.path.join(person_dir, filename)
                    image = face_recognition.load_image_file(filepath)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        person_encodings.append(encoding[0].tolist())  # Convert encoding to list
            
            if person_encodings:
                known_faces[person_name] = person_encodings

    # Save the encodings and names in JSON format
    with open(encodings_file, 'w') as file:
        json.dump(known_faces, file)
    print(f"Encoded faces for {len(known_faces)} people.")

if __name__ == '__main__':
    encode_known_faces()
