from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG to see all log messages

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            # This should stream frames; simplified example
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
