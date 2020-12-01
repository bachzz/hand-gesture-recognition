from __future__ import print_function
import cv2
import base64
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

alphabet =["A","B","C","D","E","F","G","H","I","J","K","L","M",
            "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# Test every character in the alphabet
for i in range(len(alphabet)):
    character = alphabet[i]
    path = os.getcwd()
    path = os.path.join(path, "data", character)

    base64_list = []
    for image in os.listdir(path):
        image_path = os.path.join(path, image)
        img = cv2.imread(image_path)
        img_encoded = cv2.imencode('.jpg', img)[1]
        img_base64 = base64.b64encode(img_encoded)
        img_base64 = img_base64.decode('utf-8')
        # print(img_base64)
        base64_list.append(img_base64)

    return_value = [] # The base64 for all image of a character

    # Make request to server and receive prediction result
    for j in range(len(base64_list)):
        url = 'http://127.0.0.1:5000/recognize'  # Set destination URL here

        post_fields = {'image': base64_list[j]}  # Set POST fields here

        request = Request(url, urlencode(post_fields).encode())
        json = urlopen(request, timeout=3000).read().decode() # Prediction
        return_value.append(json)
        print(json)
    # File to store the predict result of each character
    result_file = os.path.join(os.getcwd(), "test_result", character + ".txt")
    with open(result_file, 'w') as f:
        for item in return_value:
            f.write("%s\n" % item)
