# Loading Data

import pandas as pd 
from skimage.io import imread
import numpy as np

def read_data(typeData, labelsInfo, imageSize, path):
 #Intialize x  matrix
 x = np.zeros((labelsInfo.shape[0], imageSize))

 for (index, idImage) in enumerate(labelsInfo["ID"]):
  #Read image file
  nameFile = "{0}/{1}Resized/{2}.Bmp".format(path, typeData, idImage)
  img = imread(nameFile, as_grey=True)

  x[index, :] = np.reshape(img, (1, imageSize))
 return x

imageSize = 400 # 20 x 20 pixels

#Set location of data files , folders
path = ...

labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(path))

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

#Read information about test data ( IDs ).
labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(path))

#Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)

yTrain = map(ord, labelsInfoTrain["Class"])

# Defining main functions

def euclidean_distance (a, b):
 dif = a - b
 return dif.dot(dif)

def get_k_nearest_neighbors(x, i, k):
 imageI = x[i,:]
 distances = [euclidean_distance(imageI, x[j,:]) for j in xrange(x.shape[0])]     
 sortedNeighbors = np.argsort(distances)
 kNearestNeighbors = sortedNeighbors[1:(k+1)]
 return kNearestNeighbors

def assign_label(x, y, k, i):
 kNearestNeighbors = get_k_nearest_neighbors(x, i, k)
 counts = {}
 highestCount = 0
 mostPopularLabel = 0
 for n in kNearestNeighbors:
  labelOfN = y[n]
  if labelOfN not in counts :
   counts[labelOfN] = 0
  counts[labelOfN] += 1
  if counts[labelOfN] > highestCount :
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN
 return mostPopularLabel

# Running LOOF-CV with 1NN sequentially

import time
start = time.time()
k=1
yPredictions = [assign_label(xTrain, yTrain, k, i) for i in xrange(xTrain.shape[0])]
print time.time() - start, "seconds elapsed"

