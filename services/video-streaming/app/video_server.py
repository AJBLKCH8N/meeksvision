from flask import Flask, Response
from stream_handler import frame_queue
import cv2
import logging

app = Flask(__name__)

# Configure logging for Flask
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate():
    while True:
        try:
            frame = frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            logging.debug("Streaming frame.")
        except Exception as e:
            logging.error(f"Failed to encode or stream frame: {e}")

@app.route('/video_feed')
def video_feed():
    try:
        logging.info("Video feed requested.")
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error(f"Error streaming video feed: {e}")
        return Response(status=500)

def run_video_server():
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
        logging.info("Video streaming server running.")
    except Exception as e:
        logging.error(f"Failed to run the video server: {e}")
