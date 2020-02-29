import os
from scipy.io import loadmat,savemat
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
import pandas as pd
from sklearn.linear_model import Ridge
import sys

relativeCurrDir = sys.argv[1]
inputFile = relativeCurrDir+"/InputStaticFeature.csv"
outputFile = relativeCurrDir+"/OutputStaticFeature.csv"

DataInput = pd.read_csv(inputFile, sep=',',header=None)
DataOutput = pd.read_csv(outputFile, sep=',',header=None)
filename = relativeCurrDir+'/Trained_Weights.sav'

linearTrainingModel = Ridge(alpha=1.0)
linearTrainingModel.fit(DataInput, DataOutput)
r2ScoreTraining = linearTrainingModel.score(DataInput,DataOutput)

ImagesDir = relativeCurrDir+"/Images"
predictionDir = relativeCurrDir+"/PredictionRaw"

def prediction(username, num):

    if not os.path.exists(predictionDir):
        os.mkdir(predictionDir)

    PredictedImg = Image.new('1',(1920, 1200), color=0)

    Bbox = Image.open(os.path.join(ImagesDir, str(username) + "_" + str(num)+ "_bbox.png"))
    Mouse = Image.open(os.path.join(ImagesDir, str(username) + "_" + str(num)+ "_mouse.png"))
    Cursor = Image.open(os.path.join(ImagesDir, str(username) + "_" + str(num)+ "_cursor.png"))

    for y in range(2):
        for x in range(2):
            temp = []
            temp.append(Mouse.getpixel(( x, y)))
            temp.append(Cursor.getpixel(( x, y)))
            temp.append(Bbox.getpixel(( x, y)))
            npTemp = np.array(temp).reshape(1, -1)
            fixPred = linearTrainingModel.predict(npTemp)
            if fixPred > 5:
                PredictedImg.putpixel((x,y), 1)
            else:
                PredictedImg.putpixel((x,y), 0)

    PredictedImg.save(os.path.join(predictionDir, str(username) + "_Prediction_"+str(num)+".png"))


prediction('Srikanth', 14)
prediction('Varun', 14)
