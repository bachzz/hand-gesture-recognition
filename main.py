import numpy as np
import cv2
import pickle
import time

from keras.models import load_model
from keras.models import model_from_json

# model = load_model('mnist-model.f5')
lb = pickle.load(open("lb.h5","rb"))
json_file = open("model.json", "r")
model = model_from_json(json_file.read())
model.load_weights("model_weights.h5")
# model._make_predict_function()

img_tmp = cv2.imread("./pics/D1.jpg")
# img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
# img_tmp = cv2.resize(img_tmp, (28,28)) 
img_tmp = cv2.resize(img_tmp, (64,64)) 
# cv2.imshow('tmp',img_tmp)
img_tmp = np.expand_dims(img_tmp,axis=0)
# img_tmp = img_tmp.reshape(1,28,28,1)

cap = cv2.VideoCapture(0)


##### MNIST #####

alphabet = np.array(["A","B","C","D","E","F","G","H","I","J","K","L","M",
            "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"])
# alphabet = {
#     "A":0, "B":1, "C":2, "D":3, "E":4, "F": 5,
#     "G":6, "H":7, "I":8, "K":10, "L":11, "M":12,
#     "N":13, "O":14, "P":15, "Q":16, "R":17, "S": 18,
#     "T":19, "U":20, "V":21, "W":22, "X":23, "Y":24

#     0:"A",1:"B",
# }

print(len(alphabet))

########


##### ASL #####

# alphabet = np.array(["A","B","C","D","E","F","G","H","I","J","K","L","M",
#             "N","O","P","Q","R","S","T","U","V","W","X","Y","Z","del","nothing","space"])
# print(len(alphabet))

##############

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    
    x, y, w, h = 100, 100, 200, 200

    color = (255, 0, 0)
    thickness = 2

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness) 

    imgCrop = img[y:y+h, x:x+w]
    imgCrop = cv2.resize(imgCrop, (28,28)) 
    # imgCrop = cv2.resize(imgCrop, (64,64)) 

    # imgCrop = np.expand_dims(imgCrop, axis=0)
    imgCrop = imgCrop / 255
    # print(imgCrop)
    cv2.imshow('crop',imgCrop)
    imgCrop = imgCrop.reshape(1,28,28,1)
    
    # print(img_tmp.shape)


    # print(imgCrop.shape)

    pred = model.predict(imgCrop)
    # print(pred)
    # pred = np.round(pred).astype(bool)

    # print(pred)
    # print(pred[0])
    # print(np.argmax(pred))
    print(alphabet[lb.inverse_transform(pred)[0] ])
    # print(alphabet[np.argmax(pred)])
    # print(np.ma.masked_array(alphabet, pred[0]))
    # print(imgCrop)
    # x_tmp = imgCrop.reshape(-1,28,28,1)
    # predict = model.predict(x_tmp)
    # print(lb.inverse_transform(predict)[0],alphabet[lb.inverse_transform(predict)[0]])
    # print(lb.inverse_transform(predict)[0], get_key(lb.inverse_transform(predict)[0]))

    # Display the resulting frame
    cv2.imshow('frame',img)
    # cv2.imshow('crop',imgCrop)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    time.sleep(1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()