# import the opencv library
import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model(r"C:/Users/DELL-ACXIOM-3/Downloads/final_model.h5")

# define a video capture object
vid = cv2.VideoCapture(0)
fix = 220
cls = ['crosswalk', 'exit', 'men wash room', 'slippery road', 'stair', 'stop', 'women wash room', 'bump']
c=0
while(True):
	
	# Capture the video frame
	# by frame
	ret, img = vid.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	im = cv2.resize(img, (220,220))
	im = cv2.normalize(im, np.zeros((220, 220)), 0, 255, cv2.NORM_MINMAX)
	# img = np.array(img).reshape(fix, fix, 3)
	im = np.array([im]).reshape(fix, fix, 1)
	# print(model.predict(np.array([im]), verbose = 0))
	print(cls[np.argmax(model.predict(np.array([im/255]), verbose = 0))])
	# Display the resulting frame
	cv2.imshow('frame', img)
	# cv2.imwrite(r'C:/Users/DELL-ACXIOM-3/face login Complete system/traffic_symbols/'+str(c)+'.jpg', cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
	c+=1

	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
