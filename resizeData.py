import glob
from skimage.transform import resize
from skimage.io import imread, imsave
import os

#Set path of data files
path = os.path.expanduser('~') + "/workspace/julia"

if not os.path.exists( path + "/trainResized32" ):
	os.makedirs( path + "/trainResized32" )
if not os.path.exists( path + "/testResized32" ):
	os.makedirs( path + "/testResized32" )

trainFiles = glob.glob( path + "/train/*" )
for i, nameFile in enumerate(trainFiles):
	image = imread( nameFile )
	imageResized = resize( image, (32,32) )
	newName = "/".join( nameFile.split("/")[:-1] ) + "Resized32/" + nameFile.split("/")[-1]	
	imsave ( newName, imageResized )

testFiles = glob.glob( path + "/test/*" )
for i, nameFile in enumerate(testFiles):
	image = imread( nameFile )
	imageResized = resize( image, (32,32) )	
	newName = "/".join( nameFile.split("/")[:-1] ) + "Resized32/" + nameFile.split("/")[-1]	
	imsave ( newName, imageResized )
