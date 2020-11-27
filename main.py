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


cap = cv2.VideoCapture(0)


##### MNIST #####

alphabet = np.array(["A","B","C","D","E","F","G","H","I","J","K","L","M",
            "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"])

print(len(alphabet))

count = 0

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


    ### CAPTURE DATA ###
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('s'):
        count += 1
        print("haha "+ str(count))
        outpath = "./data/F/F"+str(count)+".jpg" ### change path HERE
        cv2.imwrite(outpath, imgCrop)
    elif key_pressed == ord('q'):
        break
    ####################

    imgCrop = imgCrop / 255
    cv2.imshow('crop',imgCrop)
    imgCrop = imgCrop.reshape(1,28,28,1)

    pred = model.predict(imgCrop)
    print(alphabet[lb.inverse_transform(pred)[0] ])

    # Display the resulting frame
    cv2.imshow('frame',img)
    # cv2.imshow('crop',imgCrop)
    
    # time.sleep(1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()