# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 22:06:11 2017

@author: cai

codes for Titanic practise
"""
import numpy as np
import pandas as pd
import os
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

rootPath = "/home/cai/KaggleCodes/TitanicTrain"
trainPath = os.path.join(rootPath, "train.csv")
dataTrain = pd.read_csv(trainPath)
# show full train data format
#print(dataTrain)
# show info of trainData
#print(dataTrain.info())
# show distribution
#print(dataTrain.describe())

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

