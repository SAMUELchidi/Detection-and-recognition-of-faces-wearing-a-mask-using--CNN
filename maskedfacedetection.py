                                                                                                  
																		        	#Comments SECTION	
																			                  #Requirements 
																					                # Tensorflow, Numpy, Open cv which is imported as cv2, imutils and keras.	
																				                	# We are using the mobilenet_v2 architecture.  
																								    # We import the os which will follow the path to our dataset
																									# Import the required packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os

                                                                                            #PART I
                                                                                                #This def function makes sure we take get hold of the dimensions needed
	                                                                                            #we then use the dimensions to construct a binary large object initiated as blpob
																						    
																									
	                                                                                            # Inorder to get the facial detection;-
																								    # We are passing binary large object through the CNN  network.
                                                                                                 #It will allow us loop over the detections.
																								 #We will extract the probability of the detection and express it as confidence
																								 #We the add the boxes bounding the faces then loading our model from disq

def detect_and_predict_mask(frame, faceNet, maskNet):                                                                                                                                           
	(h, w) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))


	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)  


	faces = []
	locs = []
	preds = []


	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")


			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]


			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)


			faces.append(face)
			locs.append((startX, startY, endX, endY))         
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)


	return (locs, preds)
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")
                                                                                                              #PART II

                                                                                                    # Starting our live stream 
																									#We used an internal camera  PC webcam and a external samsung phone camera whose IP adress is http://192.168.43.1:8080/video
                                                                                                    #The live stream lets us loop over the frames
                                                                                                    #detection of the faces
                                                                                                    #We give our bounding boxes color and include probability on the frame
                                                                                                    #displaying the final frame
																									#cleanup


print("Opening Camera...loading%")            
vs = VideoStream(src=0).start()
video_capture_1 = cv2.VideoCapture(0)     

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)


	for (box, pred) in zip(locs, preds):


		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred


		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)


		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(frame, label, (startX, startY - 10),


			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()
