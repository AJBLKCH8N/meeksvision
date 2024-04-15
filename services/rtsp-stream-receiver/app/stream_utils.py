import cv2
import zmq
import numpy as np
import logging
from retry import retry

logging.basicConfig(level=logging.INFO)

@retry(tries=5, delay=2)
def connect_to_stream(url):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logging.error(f"Failed to connect to {url}")
        raise ConnectionError(f"Could not open stream {url}")

    logging.info(f"Successfully connected to {url}")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Stream disconnected: {url}")
            break
        encoded, buffer = cv2.imencode('.jpg', frame)
        socket.send(buffer.tobytes())

    cap.release()
    socket.close()
    context.term()
    logging.info(f"Connection closed for {url}")
