import numpy as np
import cv2
import logging
from face_recognition import detect_faces, recognize_faces, annotate_frame
from utils import initialize_context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        context, sub_socket, pub_socket = initialize_context()
        logging.info("Context and sockets initialized successfully.")
    except Exception as e:
        logging.error("Failed to initialize ZeroMQ context and sockets: {}".format(e))
        return
    
    while True:
        try:
            frame_bytes = sub_socket.recv()
            frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            
            # Detect and recognize faces
            face_locations, face_names = recognize_faces(frame, detect_faces(frame))
            
            # Annotate the frame
            annotated_frame = annotate_frame(frame, face_locations, face_names)

            # Encode and send the annotated frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            pub_socket.send(buffer.tobytes())
        except Exception as e:
            logging.error("Error during frame processing: {}".format(e))
            continue

if __name__ == "__main__":
    main()
