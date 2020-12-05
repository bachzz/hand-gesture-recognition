from __future__ import print_function
import cv2
import base64
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from http.client import IncompleteRead

import pickle
from keras.models import load_model
from keras.models import model_from_json
import numpy as np

alphabet =["A","B","C","D","E","F","G","H","I","_","K","L","M",
            "N","O","P","Q","R","S","T","U","V","W","X","Y"]

lb = pickle.load(open("lb.h5", "rb"))
json_file = open("model.json", "r")
model = model_from_json(json_file.read())
model.load_weights("model_weights.h5")



# Test every character in the alphabet
for i in range(len(alphabet)):
    if alphabet[i] == "_":
        continue
    character = alphabet[i]
    path = os.getcwd()
    path = os.path.join(path, "data", character)

    base64_list = []


    # File to store the predict result of each character
    result_file = os.path.join(os.getcwd(), "test_result", character + ".txt")

    for image in os.listdir(path):

        image_path = os.path.join(path, image)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # # print(img)
        # img_encoded = cv2.imencode('.png', img)[1]
        img_base64 = base64.b64encode(img)
        img_base64 = img_base64.decode('utf-8')

        # img_base64 = img_encoded.tostring()
        img_base64 = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_base64, np.uint8)
        # print(nparr)
        # tmp = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        # # print(tmp)
        # print(tmp.shape)
        tmp = cv2.resize(img, (28, 28))
        tmp = tmp.reshape((1, 28, 28, 1))
        pred = model.predict(tmp)

        pred =  alphabet[lb.inverse_transform(pred)[0]] 
        print(pred)
        # return_value.append(pred)

        with open(result_file, 'a') as f:
            f.write("%s\n" % pred)

    #     # print(img_base64)
    #     base64_list.append(img_base64)

    # return_value = [] # The base64 for all image of a character

    # # Make request to server and receive prediction result
    # for j in range(len(base64_list)):
    #     url = 'http://127.0.0.1:5000/recognize'  # Set destination URL here

    #     post_fields = {'image': base64_list[j]}  # Set POST fields here

    #     request = Request(url, urlencode(post_fields).encode())
        
    #     try:
    #         json = urlopen(request, timeout=3000).read().decode() # Prediction
    #     except IncompleteRead:
    #         continue

    #     return_value.append(json)
    #     print(json)
        
    # # File to store the predict result of each character
    # result_file = os.path.join(os.getcwd(), "test_result", character + ".txt")
    # with open(result_file, 'w') as f:
    #     for item in return_value:
    #         f.write("%s\n" % item)
