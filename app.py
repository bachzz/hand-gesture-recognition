from sys import stdout
from makeup_artist import Makeup_artist
import logging
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from camera import Camera
from utils import base64_to_pil_image, pil_image_to_base64
from flask_cors import CORS
import threading

host = '127.0.0.1'#'0.0.0.0'
port = '8000'#8080

app = Flask(__name__)
CORS(app)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app, cors_allowed_origins='*')#, async_mode='eventlet')
camera = Camera(Makeup_artist())


@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)

import time
@socketio.on('connect', namespace='/test')
def connect():
    socketio.start_background_task(target=lambda: worker())
    app.logger.info("client connected")

def worker():
    while True:
        image_data = camera.get_frame() # Do your magical Image processing here!!
        while not image_data:
            image_data = camera.get_frame()
            socketio.sleep(0.05)
        image_data = image_data.decode("utf-8")
        image_data = "data:image/jpeg;base64," + image_data
        socketio.emit('out-image-event', {'image_data': image_data}, namespace='/test')


@app.route('/')
def index():
    """Video streaming home page."""

    return render_template('index2.html')



if __name__ == '__main__':
    socketio.run(app, host=host, port=port)

