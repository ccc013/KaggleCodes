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
shopInfoPath = os.path.join(rootPath, 'shopInfo.csv')
usePayPath = os.path.join(rootPath, 'userPay.csv')
useViewPath = os.path.join(rootPath, 'userView.csv')

shop_df = pd.read_csv(shopInfoPath)
print(shop_df.info())
print(shop_df.describe())

userPay_df = pd.read_csv(usePayPath)
userView_df = pd.read_csv(useViewPath)
print(userPay_df.info())
#print(userView_df.info())

 
#==============================================================================
# sales = [0] * 2000
# for i in userPay_df['shop_id']:
#     ind = int(i)-1
#     sales[ind] += 1
#==============================================================================

#for index, val in enumerate(sales):
 #   print("%d: %d" % (index, val))

#==============================================================================
# views = [0] * 2000
# for i in userView_df['shop_id']:
#     ind = int(i)-1
#     views[ind] += 1
#==============================================================================

#for index, val in enumerate(views):
#    print("%d: %d" % (index, val))
   
#shop_df['sales'] = sales
#shop_df['views'] = views
#shop_df.to_csv('shopInfo.csv', index=False)
# 统计每家商店每天购买人数
shopId = shop_df['shop_id']
#print(shopId)
shopPay_df = pd.date_range('20150701','20161031')
s_list = [i for i in range(1,2001)]
shopPayDf = pd.DataFrame(0, index = shopPay_df, columns=s_list)
#shopView_df.insert(0, shopId)

print(shopPayDf.loc['20150701',1])

print(shopPayDf.head())
shopPayDf.to_csv('shopPay.csv', index=True)

print("begin to count peoples per shop per day...")
ignKey = dict()
for t, i in zip(userPay_df['time_stamp'],userPay_df['shop_id']):
     ts = t.split(' ')
     dateKey = ts[0].split('-')
     key = dateKey[0] + dateKey[1] + dateKey[2]
     #timeKey = ts[1].split(':')[0]
     #if key not in shopPayDf.index:
     if key not in shopPayDf.index:
         if key not in ignKey:
             ignKey[key] = 0
         ignKey[key] += 1
         continue
     shopPayDf.loc[key, i] += 1
     
     #print(key,i)
     #break
     
for key, item in ignKey.items():
    print(key, item)
     
print(shopPayDf)
print(shopPayDf.info())
print(shopPayDf.describe())

shopPayDf.to_csv('shopPay.csv', index=True)

