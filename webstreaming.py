# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
#from pyimagesearch.face_blurring import anonymize_face_pixelate
#from pyimagesearch.face_blurring import anonymize_face_simple
from imutils.video import VideoStream
import numpy as np
from flask import Response
from flask import Flask
from flask import render_template
from flask_cors import CORS, cross_origin
from keras.models import load_model
from keras.models import model_from_json
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import pickle

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

alphabet = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                     "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])

# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")


def hand_recongize(frameCount):
	global vs, outputFrame, lock

	while (True):
    # Capture frame-by-frame
		img = vs.read()
		width = 800
		height = 550
		dim = (width, height)
		img = cv2.resize((img), dim)
		x, y , w, h = 100, 100, 200, 200
		
		color = (255, 0, 0)
		thickness = 2

		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

		imgCrop = img[y:y + h, x:x + w]
		imgCrop = cv2.resize(imgCrop, (28, 28))
    	# imgCrop = cv2.resize(imgCrop, (64,64)) 

		character = 'Q'
    ### CAPTURE DATA ###
		key_pressed = cv2.waitKey(1) & 0xFF
		if key_pressed == ord('s'):
			count += 1
			print("haha " + character +str(count))
			outpath = "./data/" + character + '/' + character + str(count) + ".jpg"  ### change path HERE
			cv2.imwrite(outpath, imgCrop)
		elif key_pressed == ord('q'):
			break
    ####################

		imgCrop = imgCrop / 255	
#		cv2.imshow('crop', imgCrop)
		imgCrop = cv2.resize(imgCrop, (28, 28))
		imgCrop = imgCrop.reshape(1, 28, 28, 1)

		pred = model.predict(imgCrop)
		print(alphabet[lb.inverse_transform(pred)[0]])

		with lock:
			outputFrame = img.copy()

	


def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-fc", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	ap.add_argument("-model", "--model", required=True,
		help="path to face detector model directory")
	ap.add_argument("-b", "--blocks", type=int, default=20,
		help="# of blocks for the pixelated blurring method")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	#prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	#weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
	
	lbPath = os.path.sep.join([args["model"], "lb.h5"])
	jsonPath = os.path.sep.join([args["model"], "model.json"])
	modelWeightsPath = os.path.sep.join([args["model"], "model_weights.h5"])
    
    #net = cv2.dnn.readNet(prototxtPath, weightsPath)
	lb = pickle.load(open(lbPath, "rb"))
	json_file = open(jsonPath, "r")
	model = model_from_json(json_file.read())
	model.load_weights(modelWeightsPath)


	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
#	vs = cv2.VideoCapture(0)
	time.sleep(2.0)

	# start a thread that will perform motion detection
	t = threading.Thread(target=hand_recongize, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
cv2.destroyAllWindows()
vs.stop()
