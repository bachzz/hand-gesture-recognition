from __future__ import print_function
import requests
import json
import cv2
import base64
import os
import numpy as np

#Change this path
path = "/home/hoainam/Class-ICT/semester9/cv/hand-gesture-recognition/data/E"

for image in os.listdir(path):
    image_path = os.path.join(path, image)
    img = cv2.imread(image_path)
    img_encoded = cv2.imencode('.jpg', img)[1]
    img_base64 = base64.b64encode(img_encoded)
    img_base64 = img_base64.decode('utf-8')
    print(img_base64)
    print("\n")

#img = cv2.imread('/home/hoainam/Class-ICT/semester9/cv/hand-gesture-recognition/data/A/A428.jpg')
#print(img)
#encode image as jpeg
#img_encoded = cv2.imencode('.jpg', img)[1].tostring()
#jpg_as_text = base64.b64encode(img_encoded)
#print(jpg_as_text)
