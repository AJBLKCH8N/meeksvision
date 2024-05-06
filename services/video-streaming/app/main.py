from stream_handler import start_frame_receiver
from video_server import run_video_server

def main():
    # Start the frame receiver in a separate thread
    start_frame_receiver()

    # Start the video server
    run_video_server()

if __name__ == "__main__":
    main()
