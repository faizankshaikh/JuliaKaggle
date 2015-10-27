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
yTrain = np.array(yTrain)

# Importing main functions

from sklearn.cross_validation import cross_val_score as k_fold_CV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.grid_search import GridSearchCV

# Running LOOF-CV with 1NN sequentially

import time
start = time.time()
model = KNN(n_neighbors=1)
cvAccuracy = np.mean(k_fold_CV(model, xTrain, yTrain, cv=2, scoring="accuracy"))
print "The 2-CV accuracy of 1NN", cvAccuracy
print time.time() - start, "seconds elapsed"

# Tuning the value for k

start = time.time()
tuned_parameters = [{"n_neighbors":list(range(1,5))}]
clf = GridSearchCV( model, tuned_parameters, cv=5, scoring="accuracy")
clf.fit(xTrain, yTrain)
print clf.grid_scores_
print time.time() - start, "seconds elapsed"

