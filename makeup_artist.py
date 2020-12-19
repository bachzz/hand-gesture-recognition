import cv2
import pickle
import time
import numpy as np
from keras.models import model_from_json


lb = pickle.load(open("lb.h5", "rb"))
json_file = open("model.json", "r")
model = model_from_json(json_file.read())
model.load_weights("model_weights.h5")
import time
alphabet = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                     "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])
class Makeup_artist(object):
    def __init__(self):
        pass

    def apply_makeup(self, img):
        x, y, w, h = 25, 25, 78, 53
        color = (255, 0, 0)
        thickness = 2
        imgCrop = img[y:y + h, x:x + w]
        imgCrop = imgCrop[:,:,0]
        imgCrop = cv2.resize(imgCrop, (28, 28))
        imgCrop = imgCrop / 255
        imgCrop = imgCrop.reshape(1, 28, 28, 1)

        pred = model.predict(imgCrop)

        cv2.putText(img, alphabet[lb.inverse_transform(pred)[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return img
