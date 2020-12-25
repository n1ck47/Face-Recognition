import numpy as np
import cv2 

#Init Camera 
cap = cv2.VideoCapture(0)

# Harrcascade Classifier Object
faceCascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt.xml')

count = 0
faceData = []

datasetPath = './Faces/'

fileName = input( "Enter your name : ")

while True:

	# Capture frame
	ret,frame = cap.read()
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )

	#Adjust Histogram Estimation

	clahe = cv2.createCLAHE( 2.0 , (8,8))
	grayFrame = clahe.apply( grayFrame )
	
	# Classify face from the frame and detect coordinates
	faces = faceCascade.detectMultiScale( grayFrame , 1.3 , 5 )
	faces = sorted( faces , key = lambda f:f[2]*f[3] )

	#Create Rectangle on detected Face

	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle( grayFrame , ( x,  y) , ( x + w , y + h) , ( 0,255,0) , 2 ) 

	#Cut out Region of Interest

		offset = 10 # margin in all direction
		faceSection = grayFrame[ y - offset : y + h + offset , x - offset : x + w + offset]
		faceSection = cv2.resize( faceSection , ( 100 , 100 ) ) 

		# Collect the faceSection
		count += 1
		if count%10 == 0:
			faceData.append( faceSection )
			print( len(faceData))

	cv2.imshow(" Gray Scale " , grayFrame )

	keyPressed = cv2.waitKey(1) & 0xFF
	if keyPressed == ord('q'):
		break


# Change list to np array and reshape 

faceData = np.asarray( faceData )
faceData = faceData.reshape( ( faceData.shape[0] , -1) )

#Saving the collected faces

np.save( datasetPath + fileName + '.npy' , faceData )
print( "Saved" + datasetPath + fileName )

cap.release()
cv2.destroyAllWindows()

