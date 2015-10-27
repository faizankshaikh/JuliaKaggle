import glob
from skimage.transform import resize
from skimage.io import imread, imsave
import os

#Set path of data files
path = "..."

if not os.path.exists( path + "/trainResized" ):
	os.makedirs( path + "/trainResized" )
if not os.path.exists( path + "/testResized" ):
	os.makedirs( path + "/testResized" )

trainFiles = glob.glob( path + "/train/*" )
for i, nameFile in enumerate(trainFiles):
	image = imread( nameFile )
	imageResized = resize( image, (20,20) )	
	newName = "/".join( nameFile.split("\\")[:-1] ) + "Resized/" + nameFile.split("\\")[-1]	
	imsave ( newName, imageResized )

testFiles = glob.glob( path + "/test/*" )
for i, nameFile in enumerate(testFiles):
	image = imread( nameFile )
	imageResized = resize( image, (20,20) )	
	newName = "/".join( nameFile.split("\\")[:-1] ) + "Resized/" + nameFile.split("\\")[-1]	
	imsave ( newName, imageResized )