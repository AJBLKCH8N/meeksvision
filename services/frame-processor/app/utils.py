import zmq
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_context(pub_port="tcp://*:5560"):
    try:
        context = zmq.Context()
        # Subscriber socket to receive frames from the RTSP Stream Receiver
        sub_socket = context.socket(zmq.SUB)
        sub_socket.connect("tcp://rtsp-stream-receiver:5555")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')

        # Publisher socket to send processed frames to the Video Streaming service
        pub_socket = context.socket(zmq.PUB)
        pub_socket.bind(pub_port)

        logging.info("Sockets initialized and connected/bound to ports.")
        return context, sub_socket, pub_socket
    except zmq.ZMQError as e:
        logging.error("Failed to create or configure ZeroMQ sockets: {}".format(e))
        raise

