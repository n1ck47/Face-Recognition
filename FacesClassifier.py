import numpy as np
import os
import cv2

# Camera init
cap = cv2.VideoCapture(0)

# Haarcascade Classifier
faceCascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt.xml')

# KNN

def distance( X, X_test ):
	return np.sqrt( np.sum( ( X - X_test)**2 ) )

def knn( X, X_test , k = 5 ):
	m = X.shape[0]
	distances = []

	for i in range(m):

		ix = X[ i, :-1]
		iy = X[ i , -1]

		d = distance( ix , X_test )
		distances.append([d,iy])

		dk = sorted( distances , key = lambda x:x[0] )[:k]

		labels = np.array( dk )[: , -1]

		output = np.unique( labels, return_counts = True )
		index = np.argmax( output[1] )

		return output[0][index]

##################

faceData = []
label = []
name = {}
classid = 0

dataPath = './Faces/'

for fx in os.listdir(dataPath):
	if fx.endswith('.npy'):

		# Name List
		name[classid] = fx[:-4]

		dataItem = np.load( dataPath + fx )
		faceData.append( dataItem )

		#Craete Labels
		target = classid * np.ones( ( dataItem.shape[0] , ) )
		label.append( target )
		classid += 1


faceData = np.concatenate( faceData , axis = 0 )
label = np.concatenate( label , axis = 0 )

# print( faceData.shape , label.shape )

# Concatenate X and y into training matrix

label = np.reshape( label , ( label.shape[0] ,1 ) )
X = np.hstack( ( faceData , label ) )

# Testing 

while True:

	ret,frame = cap.read()
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )

	clahe = cv2.createCLAHE( 2.0 , ( 8 , 8) )
	grayFrame = clahe.apply( grayFrame )

	faces = faceCascade.detectMultiScale( grayFrame , 1.3 , 5)

	for face in faces:
		x,y,w,h = face

		# Extracting Region of Interest
		offset = 10
		faceSection = grayFrame[ y - offset : y + h + offset , x - offset : x + w + offset]
		faceSection = cv2.resize( faceSection , (100 ,100) )

		#prediction
		out = knn( X , faceSection.flatten() )

		#Display on screen rectangle on  face and name of person
		predName = name[int(out)]
		# print( predName )
		cv2.putText( grayFrame , predName , ( x, y - 10) , cv2.FONT_HERSHEY_SIMPLEX , 1, ( 255,0,255), 2 ,cv2.LINE_AA )
		cv2.rectangle( grayFrame,  (x, y) , ( x + w , y + h) , ( 255, 0 ,0) , 2)

	cv2.imshow(" Gray " , grayFrame )

	keyPressed = cv2.waitKey(1) & 0xFF
	if keyPressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()