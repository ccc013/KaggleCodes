# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:57:23 2017

@author: cai
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
import os
sns.set_style('whitegrid')

rootPath = "/home/cai/KaggleCodes/口碑商家客流量预测"
shopInfoPath = os.path.join(rootPath, 'shop_info.csv')
usePayPath = os.path.join(rootPath, 'user_pay.txt')
useViewPath = os.path.join(rootPath, 'user_view.txt')

shop_df = pd.read_csv('shop_info.csv')
print(shop_df.info())

userPay_df = pd.read_csv('userPay.csv')
userView_df = pd.read_csv('userView.csv')
print(userPay_df.info())
print(userView_df.info())