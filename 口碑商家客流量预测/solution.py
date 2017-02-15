# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:02:46 2017

@author: cai
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
import os
import csv
sns.set_style('whitegrid')

rootPath = "/home/cai/KaggleCodes/口碑商家客流量预测"
shopInfoPath = os.path.join(rootPath, 'shop_info.txt')
usePayPath = os.path.join(rootPath, 'user_pay.txt')
useViewPath = os.path.join(rootPath, 'user_view.txt')
# txt change to csv
#==============================================================================
# csvPath = os.path.join(rootPath, 'shop_info.csv')
# txt = csv.reader(open(shopInfoPath, "r"), delimiter = ',', escapechar='\n')
# out_csv = csv.writer(open(csvPath, 'w'))
# out_csv.writerows(txt)
#==============================================================================

def getDict(filePath):
    dic = dict()
    count = 0
    with open(filePath,'r') as r:
        for line in r.readlines():
            content = line.split(',')
            if 'shop_id' not in dic.keys():
                dic['shop_id'] = list()
            dic['shop_id'].append(content[0])
            if 'city_name' not in dic.keys():
                dic['city_name'] = list()
            dic['city_name'].append(content[1])
            if 'location_id' not in dic.keys():
                dic['location_id'] = list()
            dic['location_id'].append(content[2])
            if 'per_pay' not in dic.keys():
                dic['per_pay'] = list()
            dic['per_pay'].append(content[3])
            if 'score' not in dic.keys():
                dic['score'] = list()
            if content[4] == '':
                content[4] = 'NaN'
            dic['score'].append(content[4])
            if 'comment_cnt' not in dic.keys():
                dic['comment_cnt'] = list()
            if content[5] == '':
                content[5] = 'NaN'
            dic['comment_cnt'].append(content[5])
            if 'shop_level' not in dic.keys():
                dic['shop_level'] = list()
            if content[6] == '':
                content[6] = 'NaN'
            dic['shop_level'].append(content[6])
            if 'cate_1_name' not in dic.keys():
                dic['cate_1_name'] = list()
            if content[7] == '':
                content[7] = 'NaN'
            dic['cate_1_name'].append(content[7].strip())
            if 'cate_2_name' not in dic.keys():
                dic['cate_2_name'] = list()
            if content[8] == '':
                content[8] = 'NaN'
            dic['cate_2_name'].append(content[8].strip())
            if 'cate_3_name' not in dic.keys():
                dic['cate_3_name'] = list()
            if content[9] == '':
                content[9] = 'NaN'
            dic['cate_3_name'].append(content[9].strip())
            
            count += 1
            
    return dic

def getPayDict(filePath):
    dic = dict()
    count = 0
    with open(filePath,'r') as r:
        for line in r.readlines():
            content = line.split(',')
            if 'user_id' not in dic.keys():
                dic['user_id'] = list()
            dic['user_id'].append(content[0])
            if 'shop_id' not in dic.keys():
                dic['shop_id'] = list()
            dic['shop_id'].append(content[1])
            if 'time_stamp' not in dic.keys():
                dic['time_stamp'] = list()
            dic['time_stamp'].append(content[2].strip()) 
            count += 1
    print(count)      
    return dic

#shopDict = getDict(shopInfoPath)
#s = pd.DataFrame(shopDict.items(), 
 #                columns=['shop_id', 'city_name','location_id','per_pay',
 #                'score','comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name'])
payDict = getPayDict(usePayPath)
s = pd.DataFrame(payDict)
print(s)
print(s.info())
s.to_csv('userPay.csv', index=False)

viewDict = getPayDict(useViewPath)
sv = pd.DataFrame(viewDict)
print(sv)
print(sv.info())
sv.to_csv('userView.csv', index=False)

#shop_df = pd.read_csv('shop_info.csv')

#print(shop_df.info())
