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
#print(shop_df.describe())

# shop_level 与 sales的关系
#print(shop_df['shop_level'])
# 显示shop_level的取值个数以及每种商店的数目
print(shop_df.groupby('shop_level').sales.count())#
# plot
sns.factorplot('shop_level', 'sales', data=shop_df, size=4,aspect=3)
fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='shop_level', data=shop_df, ax=axis1)
#sns.countplot(x='sales', hue='shop_level', data=shop_df,ax=axis2)
level_perc = shop_df[['shop_level', 'sales']].groupby(['shop_level'], as_index=False).mean()
sns.barplot(x='shop_level',y='sales', data=level_perc, ax=axis3)

# per_pay 与 sales的关系
# 显示per_pay的取值个数以及每种商店的数目
#print(shop_df.groupby('per_pay').sales.count())
# plot
sns.factorplot('per_pay', 'sales', data=shop_df, size=4,aspect=3)
fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='per_pay', data=shop_df, ax=axis1)
#sns.countplot(x='sales', hue='shop_level', data=shop_df,ax=axis2)
level_perc = shop_df[['per_pay', 'sales']].groupby(['per_pay'], as_index=False).mean()
sns.barplot(x='per_pay',y='sales', data=level_perc, ax=axis3)

# score 与 sales的关系
# 显示score的取值个数以及每种商店的数目
#print(shop_df.groupby('score').sales.count())


#==============================================================================
# userPay_df = pd.read_csv(usePayPath)
# userView_df = pd.read_csv(useViewPath)
# print(userPay_df.info())
#==============================================================================
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
#==============================================================================
# shopId = shop_df['shop_id']
# #print(shopId)
# shopPay_df = pd.date_range('20150701','20161031')
# s_list = [i for i in range(1,2001)]
# shopPayDf = pd.DataFrame(0, index = shopPay_df, columns=s_list)
# #shopView_df.insert(0, shopId)
#==============================================================================

#==============================================================================
# print(shopPayDf.loc['20150701',1])
# 
# print(shopPayDf.head())
# 
# print("begin to count peoples per shop per day...")
# ignKey = dict()
# count = 0
# for t, i in zip(userPay_df['time_stamp'],userPay_df['shop_id']):
#      ts = t.split(' ')
#      dateKey = ts[0].split('-')
#      key = dateKey[0] + dateKey[1] + dateKey[2]
#      #timeKey = ts[1].split(':')[0]
#      #if key not in shopPayDf.index:
#      if key not in shopPayDf.index:
#          if key not in ignKey:
#              ignKey[key] = 0
#          ignKey[key] += 1
#          continue
#      shopPayDf.loc[key, i] += 1
#      count += 1
#      if count% 1000 == 0:
#          print("%d datas have count!" % count)
#      #print(key,i)
#==============================================================================
     #break
#print("finish counting!")  
#for key, item in ignKey.items():
#    print(key, item)
     
#print(shopPayDf)
#print(shopPayDf.info())
#print(shopPayDf.describe())

#shopPayDf.to_csv('shopPay.csv', index=True)

