import zmq
import numpy as np
import cv2
from queue import Queue
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

frame_queue = Queue()

def frame_receiver():
    try:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect("tcp://frame-processor:5560")
        socket.setsockopt_string(zmq.SUBSCRIBE, '')
        logging.info("Connected to frame-processor service.")

        while True:
            try:
                frame_data = socket.recv()
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_queue.put(frame)
                    logging.debug("Frame received and added to the queue.")
                else:
                    logging.warning("Received an empty frame.")
            except zmq.ZMQError as e:
                logging.error(f"ZeroMQ error occurred: {e}")
            except Exception as e:
                logging.error(f"Unexpected error in receiving frames: {e}")
    except Exception as e:
        logging.error(f"Could not initialize the ZMQ context or socket: {e}")

def start_frame_receiver():
    thread = threading.Thread(target=frame_receiver)
    thread.start()
    logging.info("Frame receiver thread started.")
