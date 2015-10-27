import pandas as pd
import numpy as np
from skimage.io import imread
from sklearn.cross_validation import cross_val_score as k_fold_CV
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

def read_data(typeData, labelsInfo, imageSize, path):
    x = np.zeros((labelsInfo.shape[0], imageSize))
    
    for (index, idImage) in enumerate(labelsInfo["ID"]):    
        nameFile = "{0}/{1}Resized/{2}.Bmp".format(path, typeData, idImage)
        img = imread(nameFile, as_grey = True)
        
        x[index, :] = np.reshape(img, (1, imageSize))
        
    return x
    
imageSize = 400

path = "/home/faizy//workspace/julia"

labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(path))

xTrain = read_data("train", labelsInfoTrain, imageSize, path)

labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(path))

xTest = read_data("test", labelsInfoTest, imageSize, path)

yTrain = map(ord, labelsInfoTrain["Class"])
yTrain = np.array(yTrain)


model = RandomForestClassifier(max_features = 20, criterion = "entropy", n_jobs = -1)
#cvAccuracy = np.mean(k_fold_CV(model, xTrain, yTrain, cv = 2, scoring = "accuracy"))
#print "Acc: ", cvAccuracy

tuned_param = [{"n_estimators":list([10, 50, 100, 250])}]
clf = GridSearchCV(model, tuned_param, cv = 5, scoring="accuracy", verbose = 2)
clf.fit(xTrain, yTrain)
print clf.grid_scores_
