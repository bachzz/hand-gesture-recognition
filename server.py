from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from keras.models import load_model
from keras.models import model_from_json

import numpy as np
import tensorflow as tf
import cv2
import pickle
import time
import base64

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# with tf.compat.v1.Session() as sess:
lb = pickle.load(open("lb.h5", "rb"))
json_file = open("model.json", "r")
model = model_from_json(json_file.read())
model.load_weights("model_weights.h5")

alphabet = np.array(
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
     "X", "Y", "Z"])


def convert_base64_to_image(image_base64):
    try:
        image_base64 = np.fromstring(base64.b64decode(image_base64), dtype=np.uint8)
        image_base64 = cv2.imdecode(image_base64, cv2.IMREAD_GRAYSCALE)
    except:
        return None
    return image_base64


@app.route('/recognize', methods=['POST'])
@cross_origin(origin='*')
def hand_gesture_recognize():
    image_base64 = request.form.get('image')

    print(image_base64)
    #    image_base64 = convert_base64_to_image(image_base64)
    image_base64 = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
    print(image_base64)
    image_base64 = cv2.imdecode(image_base64, cv2.IMREAD_GRAYSCALE)

    print(image_base64.shape)
    # Resize image to 28*28(maybe should let client do this part)
    image_base64 = cv2.resize(image_base64, (28, 28))
    image_base64 = image_base64.reshape((1, 28, 28, 1))
    #    with tf.compat.v1.Session() as sess:
    pred = model.predict(image_base64)

    print(pred)
    return_value = alphabet[lb.inverse_transform(pred)[0]]

    return (return_value)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)
