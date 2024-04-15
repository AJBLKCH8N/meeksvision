import json
import threading
from stream_utils import connect_to_stream

def load_config():
    with open('/app/config/config.json', 'r') as config_file:
        return json.load(config_file)

def main():
    config = load_config()
    threads = []
    for url in config['rtsp_urls']:
        thread = threading.Thread(target=connect_to_stream, args=(url,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
