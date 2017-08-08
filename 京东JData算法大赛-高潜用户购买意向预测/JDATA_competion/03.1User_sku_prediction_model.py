#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2017-05-26

训练集和模型生成
"""

from collections import Counter
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os
import sys
import math
import xgboost as xgb
import numpy as np
import matplotlib.pylab as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

#sys.path.append('/home/cai/LightGBM/python-package')
import lightgbm as lgb


def age2num_2(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return 0 #3个年龄缺失的，因为很少，所以干脆当成'-1 unknown'类型
    
def clean_user_data(JData_User_Path):
    users = pd.read_csv(JData_User_Path,encoding='gbk')
    users['age'] = users['age'].map(age2num_2)
    #抽取注册时间特征,并处理缺失值
    users["user_reg_tm"] = pd.to_datetime(users["user_reg_tm"])
    users['register_year'] = users['user_reg_tm'].apply(lambda x: x.year)
    #方法一　这里使用　２０１５填充缺失值　
    #根据统计知道　users['register_year']==-1　对应的　会员等级是１　选取会员等级的众数对缺失值进行填充
    register_year_mask = users['register_year']==-1
    users.loc[register_year_mask,'register_year'] = 2015
    #去除　user_reg_tm　字段action_type6 = action[action['type']==6]

    users.drop('user_reg_tm',axis=1,inplace=True)
    return users

# ori_user_feat 修改版本V2 将 reg_year_feat 由onehot 改为 有序数值型
# 基础用户特征抽取　包含　性别　年龄　会员等级　注册时间
# 传入清洗过的　users　数据
def ori_user_feat(users):
    age_feat = pd.get_dummies(users["age"], prefix="age")
    sex_feat = pd.get_dummies(users["sex"], prefix="sex")
    #抽取注册时间特征
    reg_year_feat = pd.get_dummies(users["register_year"], prefix="reg_year")
    register_year_mapping = {
        2003:14,
        2004:13,
        2005:12,
        2006:11,
        2007:10,
        2008:9,
        2009:8,
        2010:7,
        2011:6,
        2012:5,
        2013:4,
        2014:3,
        2015:2,
        2016:1
    }
    users['register_year'] = users['register_year'].map(register_year_mapping)
    #luser_level_feat = pd.get_dummies(users["user_lv_cd"], prefix="user_level")
    #axis=1 按照列合并
    user_feat = pd.concat([users, age_feat, sex_feat], axis=1)
    #已经对['age','sex']进行过编码了，所以直接删除多余的原始['age','sex']数据列　axis=1action_type6 = action[action['type']==6]

    user_feat.drop(['age','sex'],axis=1,inplace=True)
    return user_feat

# 行为数据 生成　行为数据涉及的全部商品数目action_product.csv
def gen_all_product_from_action(all_action):
    path = "./tmp/action_product.csv"
    if os.path.exists(path):
        print "There is action_product.csv!!!"
    else:
        action = all_action[['sku_id','cate','brand']]    
        action = action.drop_duplicates('sku_id')
        print "the number of sku_id:" , action.shape[0]
        action.to_csv('./tmp/action_product.csv', index=False, index_label=False)
        print "save ./tmp/action_product.csv"
        return 
# 载入　商品数目all_product.csv
def load_all_product():
    path = "./tmp/action_product.csv"
    all_product = pd.read_csv(path)
    return all_product

def gen_all_product_feat():
    action_product = load_all_product()
    cate_feat= pd.get_dummies(action_product["cate"], prefix="cate")
    all_product_feat = pd.concat([action_product, cate_feat], axis=1)
    all_product_feat.drop('cate',inplace=True,axis=1)
    all_product_feat.drop('brand',inplace=True,axis=1)
    return all_product_feat

def ori_product_feat():
    JData_Product_Path = "./JData/JData_Product.csv"
    product = pd.read_csv(JData_Product_Path)
    a1_feat= pd.get_dummies(product["a1"], prefix="a1")
    a2_feat = pd.get_dummies(product["a2"], prefix="a2")
    a3_feat = pd.get_dummies(product["a3"], prefix="a3")
    product = pd.concat([product[['sku_id', 'brand']], a1_feat, a2_feat, a3_feat], axis=1)
    product.drop('brand',inplace=True,axis=1)
    return product

def gen_comment_feat(end_date):
    Path01 = "./JData/JData_Comment.csv"
    comment = pd.read_csv(Path01)
    comment = comment[comment.dt < end_date]
    comment_product=comment.drop_duplicates('sku_id','last')
    comment_product.drop('dt',inplace = True,axis=1)
    print "combine_product_comment enddate :",end_date
    return comment_product

# gen_user_product_table函数 是 特征组合的基础
# 考察期内出现的 用户-商品对
def gen_user_product_table(all_action,start_date, end_date):
    action = all_action[(all_action.time >= start_date) & (all_action.time < end_date)]
    action = action[['user_id', 'sku_id','type']]
    action = action.groupby(['user_id', 'sku_id'], as_index=False).sum()
    del action['type']
    print "考察期内出现的 用户-商品对",action.shape
    return action

# 生成的用户特征
# 创建当前用户总体特征
def user_active_all_statistics(user_table,action_data,end_date,flag):
    
    folder = ['train','test']
    path = "./tmp/%s/user_active_all_statistics%s.csv" % (folder[flag],end_date)
    
    if os.path.exists(path):
        print "There is csv!!!",path
        user_table  = pd.read_csv(path)
        return user_table
    
    start_date = datetime.strftime((datetime.strptime(end_date,"%Y-%m-%d")-timedelta(days=15)),"%Y-%m-%d")
    action_data = action_data[(action_data.time>=start_date)&(action_data.time<end_date)]
    #以天数为最小单位
    action_data_day = action_data[['user_id', 'time','userLook','userAdd','userDelete','userBuy','userFavor','userClick']]
#时间计算
    print "任意操作"
    #任意操作
    #【最早操作时间】【最晚操作时间】【最早和最晚时间间隔】【有操作的天数】【有操作的天数/最早最晚时间间隔】
    user_day_action = action_data_day.groupby(
        ['user_id','time']).sum().reset_index()
    user_day_action = user_day_action.sort(
        ['user_id','time'],ascending=True).reset_index()
    time = user_day_action.groupby(['user_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    #print time
    time['userFirstActiveDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['userLastActiveDay'] = time['time'].apply(
        lambda x: (x.split(':'))[len(x.split(':'))-1])
    time['userFirstActiveDay'] = map(
        lambda x:datetime.strptime(x,"%Y-%m-%d"),time['userFirstActiveDay'])
    time['userLastActiveDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['userLastActiveDay'])
    time['userFirstLastActiveDaysDifference'] = time['userLastActiveDay']-time['userFirstActiveDay']
    time['userFirstLastActiveDaysDifference'] = map(
        lambda x:x/np.timedelta64(1,'D')+1,time['userFirstLastActiveDaysDifference'])
    
    time['userFirstActiveDay'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['userFirstActiveDay'])
    time['userLastActiveDay'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['userLastActiveDay'])
    time['userActiveDays'] = time['time'].apply(
        lambda x: len(x.split(':')))
    time['userActiveDaysRatio'] = time['userActiveDays']/time['userFirstLastActiveDaysDifference']
    time = time.drop('time',1)
    user_table =pd.merge(user_table,time,how='left',on='user_id')
    Nan_LastActiveDay = (datetime.strptime(end_date,"%Y-%m-%d")-datetime.strptime(start_date,"%Y-%m-%d")).days+1
    Nan_FirstActiveDay = -1
    Nan_FirstLastActiveDaysDifference = -1
    Nan_ActiveDays = 0
    Nan_AciveDaysRatio = 0 
    user_table['userLastActiveDay'] = user_table['userLastActiveDay'].fillna(Nan_LastActiveDay)
    user_table['userFirstActiveDay'] = user_table['userFirstActiveDay'].fillna(Nan_FirstActiveDay)
    user_table['userFirstLastActiveDaysDifference'] = user_table['userFirstLastActiveDaysDifference'].fillna(Nan_FirstLastActiveDaysDifference)
    user_table['userActiveDays'] = user_table['userActiveDays'].fillna(Nan_ActiveDays)
    user_table['userActiveDaysRatio'] = user_table['userActiveDaysRatio'].fillna(Nan_AciveDaysRatio)
    # print time
    print "部分操作"
    #部分操作
    #【最早有购买或购买意向的时间】【最晚有购买和购买意向的时间】【有购买意向的天数/最早最晚时间间隔】
    user_day_buyintension = (action_data_day[
        (action_data_day.userAdd==1)|(action_data_day.userBuy==1)
        |(action_data_day.userFavor==1)]).groupby(
        ['user_id', 'time']).sum().reset_index()
    user_day_buyintension = user_day_buyintension.sort(
        ['user_id', 'time'], ascending=True).reset_index()
    time_buyintension = user_day_buyintension.groupby(['user_id'])['time'].agg(
        lambda x: ':'.join(x)).reset_index()
    time_buyintension['userFirstActiveDayBuyIntension'] = time_buyintension['time'].apply(
        lambda x: (x.split(':'))[0])
    time_buyintension['userLastActiveDayBuyIntension'] = time_buyintension['time'].apply(
        lambda x: (x.split(':'))[len(x.split(':')) - 1])
    time_buyintension['userFirstActiveDayBuyIntension'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"),
        time_buyintension['userFirstActiveDayBuyIntension'])
    time_buyintension['userLastActiveDayBuyIntension'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"),
        time_buyintension['userLastActiveDayBuyIntension'])
    time_buyintension['userFirstActiveDayBuyIntension'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d") - x).days,
        time_buyintension['userFirstActiveDayBuyIntension'])
    time_buyintension['userLastActiveDayBuyIntension'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d") - x).days,
        time_buyintension['userLastActiveDayBuyIntension'])
    time_buyintension['userActiveDaysBuyIntension'] = time_buyintension['time'].apply(
        lambda x: len(x.split(':')))
    time_buyintension = pd.merge(time_buyintension,time[['user_id','userFirstLastActiveDaysDifference','userActiveDays']],how='left',on='user_id')
    time_buyintension['userActiveDaysBuyIntensionRatio'] = \
                     time_buyintension['userActiveDaysBuyIntension'] / time_buyintension['userFirstLastActiveDaysDifference']
    time_buyintension = time_buyintension.drop(['time','userFirstLastActiveDaysDifference','userActiveDays'],1)
    user_table = pd.merge(user_table, time_buyintension, how='left', on='user_id')
    user_table['userLastActiveDayBuyIntension'] = user_table['userLastActiveDayBuyIntension'].fillna(Nan_LastActiveDay)
    user_table['userFirstActiveDayBuyIntension'] = user_table['userFirstActiveDayBuyIntension'].fillna(Nan_FirstActiveDay)
    user_table['userActiveDaysBuyIntension'] = user_table['userActiveDaysBuyIntension'].fillna(Nan_ActiveDays)
    user_table['userActiveDaysBuyIntensionRatio'] = user_table['userActiveDaysBuyIntensionRatio'].fillna(Nan_AciveDaysRatio)
    #单一操作
    # 【最早购买的时间】【最晚购买的时间】【购买的天数/最早最晚时间间隔】【购买的天数/活跃的天数】
    user_day_buy = (action_data_day[(action_data_day.userBuy == 1)]).groupby(
        ['user_id', 'time']).sum().reset_index()
    user_day_buy = user_day_buy.sort(
        ['user_id', 'time'], ascending=True).reset_index()
    time_buy = user_day_buy.groupby(['user_id'])['time'].agg(
        lambda x: ':'.join(x)).reset_index()

    time_buy['userFirstActiveDayBuy'] = time_buy['time'].apply(
        lambda x: (x.split(':'))[0])
    time_buy['userLastActiveDayBuy'] = time_buy['time'].apply(
        lambda x: (x.split(':'))[len(x.split(':')) - 1])
    time_buy['userFirstActiveDayBuy'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"),
        time_buy['userFirstActiveDayBuy'])
    time_buy['userLastActiveDayBuy'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"),
        time_buy['userLastActiveDayBuy'])
    time_buy['userFirstActiveDayBuy'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d") - x).days,
        time_buy['userFirstActiveDayBuy'])
    time_buy['userLastActiveDayBuy'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d") - x).days,
        time_buy['userLastActiveDayBuy'])
    time_buy['userActiveDaysBuy'] = time_buy['time'].apply(
        lambda x: len(x.split(':')))
    time_buy = pd.merge(time_buy,time[['user_id','userFirstLastActiveDaysDifference','userActiveDays']],how='left',on='user_id')
    time_buy['userActiveDaysBuyRatio'] = time_buy['userActiveDaysBuy'] / time_buy['userFirstLastActiveDaysDifference']
    time_buy['userAcitveBuyActiveInteractiveDaysRatio'] = time_buy['userActiveDaysBuy'] / time_buy['userActiveDays']
    time_buy = time_buy.drop(['time','userFirstLastActiveDaysDifference','userActiveDays'],1)
    user_table = pd.merge(user_table, time_buy, how='left', on='user_id')
    user_table['userLastActiveDayBuy'] = user_table['userLastActiveDayBuy'].fillna(Nan_LastActiveDay)
    user_table['userFirstActiveDayBuy'] = user_table['userFirstActiveDayBuy'].fillna(Nan_FirstActiveDay)
    user_table['userActiveDaysBuy'] = user_table['userActiveDaysBuy'].fillna(Nan_ActiveDays)
    user_table['userActiveDaysBuyRatio'] = user_table['userActiveDaysBuyRatio'].fillna(Nan_AciveDaysRatio)
 
    #操作次数计算
    actionTimes = action_data_day.groupby(['user_id']).sum().reset_index()
    # print actionTimes
    #任意操作
    #【操作总次数】
    actionTimes['userActionNum'] = actionTimes['userLook']+actionTimes['userAdd']+actionTimes['userDelete']+\
               actionTimes['userBuy']+actionTimes['userFavor']+actionTimes['userClick']
    #部分操作
    # 【购买意向总次数】
    actionTimes['userBuyIntentionNum'] = actionTimes['userAdd']+ actionTimes['userBuy']+actionTimes['userFavor']
    #【前期浏览操作总次数】
    actionTimes['userLookWaitNum'] = actionTimes['userLook']+ actionTimes['userClick']
    #【购买意向/操作比】
    actionTimes['userBuyIntentionActionRatio'] = actionTimes['userBuyIntentionNum']/actionTimes['userActionNum']
    actionTimes['userBuyIntentionActionRatio'] = actionTimes['userBuyIntentionActionRatio'].apply(lambda x:1 if math.isinf(x) else x)
    #【购买意向/前期浏览操作比】
    actionTimes['userBuyIntentionLookWaitRatio'] = actionTimes['userBuyIntentionNum'] / actionTimes['userLookWaitNum']
    actionTimes['userBuyIntentionLookWaitRatio'] = actionTimes['userBuyIntentionLookWaitRatio'].apply(lambda x:100 if math.isinf(x) else x)
    #【前期操作/操作比】
    actionTimes['userLookWaitActionRatio'] = actionTimes['userLookWaitNum'] / actionTimes['userActionNum']
    actionTimes['userLookWaitActionRatio'] = actionTimes['userLookWaitActionRatio'].apply(lambda x:1 if math.isinf(x) else x)
    print "单一操作"     
    #单一操作
    #【各操作/操作比】
    actionTimes['userLookActionRatio'] = actionTimes['userLook']/actionTimes['userActionNum']
    actionTimes['userLookActionRatio'] = actionTimes['userLookActionRatio'].apply(lambda x:1 if math.isinf(x) else x) 
    actionTimes['userAddActionRatio'] = actionTimes['userAdd'] / actionTimes['userActionNum']
    actionTimes['userAddActionRatio'] = actionTimes['userAddActionRatio'].apply(lambda x:1 if math.isinf(x) else x)   
    actionTimes['userDeleteActionRatio'] = actionTimes['userDelete'] / actionTimes['userActionNum']
    actionTimes['userDeleteActionRatio'] = actionTimes['userDeleteActionRatio'].apply(lambda x:1 if math.isinf(x) else x)   
    actionTimes['userBuyActionRatio'] =  actionTimes['userBuy'] / actionTimes['userActionNum']
    actionTimes['userBuyActionRatio'] =  actionTimes['userBuyActionRatio'].apply(lambda x:1 if math.isinf(x) else x)
    actionTimes['userFavorActionRatio'] = actionTimes['userFavor'] / actionTimes['userActionNum']
    actionTimes['userFavorActionRatio'] = actionTimes['userFavorActionRatio'].apply(lambda x:1 if math.isinf(x) else x)
    actionTimes['userClickActionRatio'] = actionTimes['userClick'] / actionTimes['userActionNum']
    actionTimes['userClickActionRatio'] = actionTimes['userClickActionRatio'].apply(lambda x:1 if math.isinf(x) else x)
    #【购买/加购比】【购买/收藏比】【购买/点击比】【购买/浏览比】
    actionTimes['userBuyAddRatio'] = actionTimes['userBuy'] / actionTimes['userAdd']
    actionTimes['userBuyAddRatio'] = actionTimes['userBuyAddRatio'].apply(lambda x:1 if (math.isinf(x))|(x>1) else x) 
    actionTimes['userBuyFavorRatio'] = actionTimes['userBuy'] / actionTimes['userFavor']
    actionTimes['userBuyFavorRatio'] = actionTimes['userBuyFavorRatio'].apply(lambda x:1 if (math.isinf(x))|(x>1) else x) 
    actionTimes['userBuyClickRatio'] = actionTimes['userBuy'] / actionTimes['userClick']
    actionTimes['userBuyClickRatio'] = actionTimes['userBuyClickRatio'].apply(lambda x:1 if (math.isinf(x))|(x>1) else x) 
    actionTimes['userBuyLookRatio'] = actionTimes['userBuy'] / actionTimes['userLook']
    actionTimes['userBuyLookRatio'] = actionTimes['userBuyLookRatio'].apply(lambda x:1 if (math.isinf(x))|(x>1) else x) 
    #【加购/收藏比】【加购/点击比】【加购/浏览比】
    actionTimes['userAddFavorRatio'] = actionTimes['userAdd'] / actionTimes['userFavor']
    actionTimes['userAddFavorRatio'] = actionTimes['userAddFavorRatio'].apply(lambda x:100 if math.isinf(x) else x) 
    actionTimes['userAddClickRatio'] = actionTimes['userAdd'] / actionTimes['userClick']
    actionTimes['userAddClickRatio'] = actionTimes['userAddClickRatio'].apply(lambda x:100 if math.isinf(x) else x) 
    actionTimes['userAddLookRatio'] = actionTimes['userAdd'] / actionTimes['userLook']
    actionTimes['userAddLookRatio'] = actionTimes['userAddLookRatio'].apply(lambda x:100 if math.isinf(x) else x) 
    #【收藏/点击比】【收藏/浏览比】
    actionTimes['userFavorClickRatio'] = actionTimes['userFavor'] / actionTimes['userClick']
    actionTimes['userFavorClickRatio'] = actionTimes['userFavorClickRatio'].apply(lambda x:100 if math.isinf(x) else x)    
    actionTimes['userFavorLookRatio'] = actionTimes['userFavor'] / actionTimes['userLook']
    actionTimes['userFavorLookRatio'] = actionTimes['userFavorLookRatio'].apply(lambda x:100 if math.isinf(x) else x) 
    #【点击/浏览比】
    actionTimes['userClickLookRatio'] = actionTimes['userClick'] / actionTimes['userLook']
    actionTimes['userClickLookRatio'] = actionTimes['userClickLookRatio'].apply(lambda x:100 if math.isinf(x) else x) 
    #【删购/加购比】
    actionTimes['userDeleteAddRatio'] = actionTimes['userDelete'] / actionTimes['userAdd']
    actionTimes['userDeleteAddRatio'] = actionTimes['userDeleteAddRatio'].apply(lambda x:100 if math.isinf(x) else x) 
    #【平均每次购买商品数】=【购买总量】/【购买天数】
    actionTimes = pd.merge(actionTimes,time_buy[['user_id','userActiveDaysBuy']],how='left',on='user_id')
    actionTimes['userAverageBuyNumber'] = actionTimes['userBuy'] / actionTimes['userActiveDaysBuy']
    actionTimes = actionTimes.drop(['userActiveDaysBuy'],axis=1)
    user_table = pd.merge(user_table,actionTimes,how='left',on='user_id')
    user_table = user_table.fillna(0)
    user_table.to_csv(path, index=False, index_label=False)
    
    return user_table

# 创建当前用户对单个的商品 的统计特征(粒度是商品,不是具体某一确定商品) 
def user_active_statistics_product(user_table,action_data,end_date,flag):
    
    folder = ['train','test']
    path = "./tmp/%s/user_active_statistics_product%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_table  = pd.read_csv(path)
        return user_table
    
    start_date = '2016-02-01'
    action_data = action_data[
        (action_data.time >= start_date) & (action_data.time < end_date)]

    #用户对交互过得商品的平均交互数，最大交互数，最小交互数
    product_interactive = action_data[['user_id', 'sku_id']]
    product_interactive['activeNum'] = 1
    product_interactive = product_interactive.groupby(
        ['user_id','sku_id']).sum().reset_index()
    product_interactive['activeNum'] = product_interactive['activeNum'].astype('str')
    product_interactive = product_interactive.groupby(
        ['user_id'])['activeNum'].agg(lambda x:':'.join(x)).reset_index()
    product_interactive['userMeanInteractive'] = product_interactive.activeNum.apply(
        lambda x: np.mean([int(d) for d in x.split(':')]))
    product_interactive['userMinInteractive'] = product_interactive.activeNum.apply(
        lambda x: min([int(d) for d in x.split(':')]))
    product_interactive['userMaxInteractive'] = product_interactive.activeNum.apply(
        lambda x: max([int(d) for d in x.split(':')]))
    product_interactive = product_interactive.drop(['activeNum'],1)
    user_table = pd.merge(user_table, product_interactive, how='left', on='user_id')
    user_table = user_table.fillna(0)
    # 用户对交互过得商品的平均购买数，最大购买数
    product_buy = (action_data[action_data.type==4])[['user_id', 'sku_id']]
    product_buy['buyNum'] = 1
    product_buy = product_buy.groupby(
        ['user_id', 'sku_id']).sum().reset_index()
    product_buy['buyNum'] = product_buy['buyNum'].astype('str')
    product_buy = product_buy.groupby(
        ['user_id'])['buyNum'].agg(lambda x: ':'.join(x)).reset_index()
    product_buy['userMeanBuy'] = product_buy.buyNum.apply(
        lambda x: np.mean([int(d) for d in x.split(':')]))
    product_buy['userMaxBuy'] = product_buy.buyNum.apply(
        lambda x: max([int(d) for d in x.split(':')]))
    product_buy = product_buy.drop(['buyNum'],1)
    user_table = pd.merge(user_table, product_buy, how='left', on='user_id')
    user_table = user_table.fillna(0)
    #用户购买过的商品数量
    product_buy_num = (action_data[action_data.type == 4])[['user_id', 'sku_id']]
    product_buy_num = product_buy_num.drop_duplicates()
    product_buy_num['userBuyProductNum'] = 1
    product_buy_num = product_buy_num[['user_id','userBuyProductNum']].groupby(
        ['user_id']).sum().reset_index()
    user_table = pd.merge(user_table, product_buy_num, how='left', on='user_id')
    user_table = user_table.fillna(0)
    # 用户重复购买的次数，重复购买的商品总数，重复购买率
    product_buyrepeat = (action_data[action_data.type == 4])[['user_id', 'sku_id']]
    product_buyrepeat['buyNum'] = 1
    product_buyrepeat = product_buyrepeat.groupby(
        ['user_id', 'sku_id']).sum().reset_index()

    #重复购买的商品总数
    product_buyrepeatnum= product_buyrepeat[product_buyrepeat.buyNum > 1]
    product_buyrepeatnum['userBuyProductsRepeat'] = 1
    product_buyrepeatnum = product_buyrepeatnum[
        ['user_id', 'userBuyProductsRepeat']].groupby(['user_id']).sum().reset_index()
    user_table = pd.merge(user_table, product_buyrepeatnum, how='left', on='user_id')
    # 重复购买的次数
    product_buyrepeat['userBuyTimeRepeat'] = map(
        lambda x:x-1, product_buyrepeat['buyNum'])
    product_buyrepeattime = product_buyrepeat[['user_id','userBuyTimeRepeat']].groupby(
        ['user_id']).sum().reset_index()
    user_table = pd.merge(user_table, product_buyrepeattime, how='left', on='user_id')
    # 重复购买率
    product_buyrepeatratio = product_buyrepeatnum[['user_id','userBuyProductsRepeat']]
    product_buyrepeatratio = pd.merge(product_buyrepeatratio,product_buy_num[['user_id','userBuyProductNum']],how='left',on='user_id')
    product_buyrepeatratio['userBuyProductRepeatRatio'] = \
                    product_buyrepeatratio['userBuyProductsRepeat']/product_buyrepeatratio['userBuyProductNum']
    product_buyrepeatratio = product_buyrepeatratio.drop(['userBuyProductsRepeat','userBuyProductNum'],1)
    user_table = pd.merge(user_table, product_buyrepeatratio, how='left', on='user_id')
    
    user_table.to_csv(path, index=False, index_label=False)
    return user_table


# 用户是否有购买记录 是否有多次购买记录
def user_buy_and_repeat_buy_flag(user_table,action_data,end_date,flag):
    
    folder = ['train','test']
    path = "./tmp/%s/user_buy_and_repeat_buy_flag%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_table  = pd.read_csv(path)
        return user_table
    print 'user_buy_and_repeat_buy_flag'
    action_data = action_data[(action_data.time<end_date)&(action_data.type==4)][['user_id','sku_id','userBuy']]
    user_is_buy = action_data.groupby(['user_id']).sum().reset_index()
    user_is_buy['userHasBoughtRecord'] = 1
    user_is_buy['userHasManyBoughtRecord'] = user_is_buy['userBuy'].apply(lambda x:1 if x>1 else 0)
    user_table=pd.merge(user_table,user_is_buy[
        ['user_id','userHasBoughtRecord','userHasManyBoughtRecord']],how='left',on='user_id')
    user_table = user_table.fillna(0)
    user_table.to_csv(path, index=False, index_label=False)
    return user_table


# 用户在截止日前 最后一个活跃日的交互商品数 和 活跃日日均交互商品数
def user_last1_activeDay_interact(user_table,action_data,end_date,flag):
    folder = ['train','test']
    path = "./tmp/%s/user_last1_activeDay_interact%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_table  = pd.read_csv(path)
        return user_table
    print 'user_last1_activeDay_interact'
    
    #活跃日日均交互商品数(排除掉315的数据和购买当天数据)
    action_active_daily_product_num = action_data[(action_data.time<end_date)&(action_data.time!='2016-03-15')][['user_id','sku_id','time','userBuy']].groupby(
    ['user_id','sku_id','time']).sum().reset_index()
    action_active_daily_product_num['productNum'] = 1
    action_active_daily_product_num = action_active_daily_product_num[['user_id','time','productNum','userBuy']].groupby(['user_id','time']).sum().reset_index()
    action_active_daily_product_num['dayNum'] = 1
    action_active_daily_product_num = action_active_daily_product_num[
        action_active_daily_product_num.userBuy==0].groupby(['user_id']).sum().reset_index()
    action_active_daily_product_num['userDailyInteractProductNum'] = action_active_daily_product_num['productNum']/action_active_daily_product_num['dayNum']
    print '活跃日日均交互商品数'
    #最后一个活跃日交互的商品数
    user_last1_activeDay_interact_product_num = action_data[action_data.time<end_date][['user_id','sku_id','time']].drop_duplicates()
    user_last1_activeDay_interact_product_num['userLast1ActiveDayInteractProductNum'] = 1
    user_last1_activeDay_interact_product_num = user_last1_activeDay_interact_product_num.groupby(
        ['user_id','time']).sum().reset_index().sort(['user_id','time'],ascending=False).drop_duplicates(['user_id'])
    print '最后一个活跃日交互的商品数'
    #合并
    user_last1_activeDay_interact_product = pd.merge(
        user_last1_activeDay_interact_product_num[['user_id','userLast1ActiveDayInteractProductNum']],
        action_active_daily_product_num[['user_id','userDailyInteractProductNum']],how='left',on='user_id')
    user_last1_activeDay_interact_product['userLast1ActiveDayInteractProductRatio'] = (user_last1_activeDay_interact_product['userLast1ActiveDayInteractProductNum'] - user_last1_activeDay_interact_product['userDailyInteractProductNum'])/ user_last1_activeDay_interact_product['userDailyInteractProductNum']
    user_table=pd.merge(user_table,user_last1_activeDay_interact_product,how='left',on='user_id')
    user_table = user_table.fillna(0)
    
    user_table.to_csv(path, index=False, index_label=False)
    return user_table


# 提取用户预测期前最后一条交互记录相关的特征
def user_last_interact_record(user_table,action_data,end_date,flag):
    folder = ['train','test']
    path = "./tmp/%s/user_last_interact_record%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_table  = pd.read_csv(path)
        return user_table
    print 'user_last_interact_record'
    
    #距离end_dated的4天及之前的全部考虑为近期无交互，当作缺失值处理
    start_date = datetime.strftime((datetime.strptime(end_date,"%Y-%m-%d")-timedelta(days=3)),"%Y-%m-%d")
    action = action_data[(action_data.time>=start_date)&(action_data.time<end_date)][['user_id','time','timeHour','timeMinute']]
    #选出用户最后一条交互数据
    action = action.sort(ascending=False).drop_duplicates(['user_id'])
    #【最后交互时间】
    action['userLastInteractTime'] = action['time'].apply(
        lambda x:(datetime.strptime(end_date,"%Y-%m-%d")-datetime.strptime(x,"%Y-%m-%d")).days-1)+(24-action['timeHour']-1)/24+(60-action['timeMinute'])/24/60
    #【最后交互时间是否是最后一天23点30后】
    action['userLastInteractIsLastDayAfter2330'] = action['userLastInteractTime'].apply(lambda x:1 if x<=30.0/60/24 else 0)
    #【最后交互时间是否是最后一天6点前】
    action['userLastInteractIsLastDayBefore0600']= action['userLastInteractTime'].apply(lambda x:1 if (x<=1.0)&(x>18.0/24) else 0)
    user_table = pd.merge(user_table,action[['user_id','userLastInteractTime','userLastInteractIsLastDayAfter2330','userLastInteractIsLastDayBefore0600']],how='left',on=['user_id'])
    #缺失值填充
    user_table['userLastInteractTime'] = user_table['userLastInteractTime'].fillna(10)
    user_table['userLastInteractIsLastDayAfter2330'] = user_table['userLastInteractIsLastDayAfter2330'].fillna(0)
    user_table['userLastInteractIsLastDayBefore0600'] = user_table['userLastInteractIsLastDayBefore0600'].fillna(0)
    user_table.to_csv(path, index=False, index_label=False)
    return user_table

def user_ActiveDay_daily_time_num(user_table,action_data,end_date,flag):
    folder = ['train','test']
    path = "./tmp/%s/user_ActiveDay_daily_time_num%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_table  = pd.read_csv(path)
        return user_table
    print 'user_ActiveDay_daily_time_num'
    
    #活跃日日均交互时长（分钟）
    action_active_daily_time_num = action_data[(action_data.time<end_date)&(action_data.time!='2016-03-15')][
        ['user_id','time','timeHour','timeMinute','userBuy']].groupby(
        ['user_id','time','timeHour','timeMinute']).sum().reset_index()
    action_active_daily_time_num['minuteNum'] = 1
    action_active_daily_time_num = action_active_daily_time_num[['user_id','time','minuteNum','userBuy']].groupby(['user_id','time']).sum().reset_index()
    action_active_daily_time_num['dayNum'] = 1
    action_active_daily_time_num = action_active_daily_time_num[
        action_active_daily_time_num.userBuy==0].groupby(['user_id']).sum().reset_index()
    action_active_daily_time_num['userDailyInteractMinuteNum'] = action_active_daily_time_num['minuteNum']/action_active_daily_time_num['dayNum'] 
    user_table = pd.merge(user_table,action_active_daily_time_num[['user_id','userDailyInteractMinuteNum']],how='left',on='user_id')
    user_table = user_table.fillna(0)
    user_table.to_csv(path, index=False, index_label=False)
    return user_table

# 用户过去N天交互的特征
def user_lastNDay_InteractTime(user_table,action_data,end_date,action_active_daily_time_num,flag,N=3):
    folder = ['train','test']
    path = "./tmp/%s/user_lastNDay_InteractTime%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_table  = pd.read_csv(path)
        return user_table
    print 'user_lastNDay_InteractTime'
    
    start_date = datetime.strftime((datetime.strptime(end_date,"%Y-%m-%d")-timedelta(days=N)),"%Y-%m-%d")
    action = action_data[(action_data.time>=start_date)&(action_data.time<end_date)][['user_id','time','timeHour','timeMinute']]
    action['minuteNum'] = 1
    action = action[['user_id','time','minuteNum']].groupby(['user_id','time']).sum().reset_index()
    #用户过去N天最活跃的天及交互时间，和日常平均时间的比值
    user_time_mostActiveDay = action.sort(['user_id','minuteNum']).drop_duplicates(['user_id'])
    user_time_mostActiveDay['userLast%dDayMostActiveDay'%N] = user_time_mostActiveDay['time'].apply(
        lambda x:(datetime.strptime(end_date,"%Y-%m-%d")-datetime.strptime(x,"%Y-%m-%d")).days)
    user_time_mostActiveDay['userLast%dDayMostActiveDayMinuteNum'%N] = user_time_mostActiveDay['minuteNum']
    user_time_mostActiveDay = pd.merge(user_time_mostActiveDay,action_active_daily_time_num,how='left',on='user_id')
    user_time_mostActiveDay['userLast%dDayMostActiveDayMinuteNumRatio'%N] = (user_time_mostActiveDay['userLast%dDayMostActiveDayMinuteNum'%N]-user_time_mostActiveDay['userDailyInteractMinuteNum'])/user_time_mostActiveDay['userDailyInteractMinuteNum']
    user_table = pd.merge(user_table,user_time_mostActiveDay[['user_id','userLast%dDayMostActiveDay'%N,'userLast%dDayMostActiveDayMinuteNum'%N,'userLast%dDayMostActiveDayMinuteNumRatio'%N]],how='left',on='user_id')
    user_table['userLast%dDayMostActiveDay'%N] = user_table['userLast%dDayMostActiveDay'%N].fillna(N+1)
    user_table = user_table.fillna(0)
    user_table['userLast%dDayMostActiveDayMinuteNumRatio'%N] = user_table['userLast%dDayMostActiveDayMinuteNumRatio'%N].apply(lambda x:0 if math.isinf(x) else x)
    #用户过去N天最活跃的天加购物车的数量（加购物车数-减购物车数）
    user_mostActive_IsAdd = action_data[(action_data.time>=start_date)&(action_data.time<end_date)][
        ['user_id','time','sku_id','userAdd','userDelete']].groupby(['user_id','time','sku_id']).sum().reset_index()
    user_mostActive_IsAdd = pd.merge(user_time_mostActiveDay[['user_id','time']],user_mostActive_IsAdd,how='inner',on=['user_id','time'])
    user_mostActive_IsAdd['userLast%dDayMostActvieDayAddNum'%N] = (user_mostActive_IsAdd['userAdd']-user_mostActive_IsAdd['userDelete']).apply(lambda x:0 if x<0 else x)
    user_mostActive_IsAdd = user_mostActive_IsAdd.groupby(['user_id','time']).sum().reset_index()
    user_table = pd.merge(user_table,user_mostActive_IsAdd[['user_id','userLast%dDayMostActvieDayAddNum'%N]],how='left',on='user_id')
    user_table = user_table.fillna(0)
    #用户过去N天最活跃的天和最后一个活跃天的差值
    user_last_active_date = action[['user_id','time']].sort(ascending=False).drop_duplicates(['user_id'])
    user_last_active_date = pd.merge(user_last_active_date,user_time_mostActiveDay[['user_id','userLast%dDayMostActiveDay'%N]],how='left',on='user_id')
    user_last_active_date['userLast%dDayLastActiveDay'%N] = user_last_active_date['time'].apply(lambda x:(datetime.strptime(end_date,"%Y-%m-%d")-datetime.strptime(x,"%Y-%m-%d")).days)
    user_last_active_date['userLast%dDayLastMostActiveDayDif'%N] = user_last_active_date['userLast%dDayMostActiveDay'%N]-user_last_active_date['userLast%dDayLastActiveDay'%N]
    user_table = pd.merge(user_table,user_last_active_date[['user_id','userLast%dDayLastActiveDay'%N,'userLast%dDayLastMostActiveDayDif'%N]],how='left',on='user_id')
    user_table['userLast%dDayLastActiveDay'%N] = user_table['userLast%dDayLastActiveDay'%N].fillna(N+1)
    user_table = user_table.fillna(0)
    #用户过去N天最后一个活跃天加购物车的数量（加购物车数-减购物车数）
    user_lastActive_IsAdd = action_data[(action_data.time>=start_date)&(action_data.time<end_date)][
        ['user_id','time','sku_id','userAdd','userDelete']].groupby(['user_id','time','sku_id']).sum().reset_index()
    user_lastActive_IsAdd = pd.merge(user_last_active_date[['user_id','time']],user_lastActive_IsAdd,how='inner',on=['user_id','time'])
    user_lastActive_IsAdd = user_lastActive_IsAdd.groupby(['user_id','time','sku_id']).sum().reset_index()
    user_lastActive_IsAdd['userLast%dDayLastActvieDayAddNum'%N] = (user_lastActive_IsAdd['userAdd']-user_lastActive_IsAdd['userDelete']).apply(lambda x:0 if x<0 else x)
    user_lastActive_IsAdd = user_lastActive_IsAdd.groupby(['user_id','time']).sum().reset_index()
    user_table = pd.merge(user_table,user_lastActive_IsAdd[['user_id','userLast%dDayLastActvieDayAddNum'%N]],how='left',on='user_id')
    user_table = user_table.fillna(0)
    #最后一个活跃日交互时长
    user_last1_activeDay_interact_time_num = action_data[action_data.time<end_date][['user_id','time','timeHour','timeMinute']].drop_duplicates()
    user_last1_activeDay_interact_time_num['userLast1ActiveDayInteractMinuteNum'] = 1
    user_last1_activeDay_interact_time_num = user_last1_activeDay_interact_time_num[['user_id','time','userLast1ActiveDayInteractMinuteNum']].groupby(
        ['user_id','time']).sum().reset_index().sort(['user_id','time'],ascending=False).drop_duplicates(['user_id'])
    user_last1_activeDay_interact_time = pd.merge(
        user_last1_activeDay_interact_time_num[['user_id','userLast1ActiveDayInteractMinuteNum']],
        action_active_daily_time_num[['user_id','userDailyInteractMinuteNum']],how='left',on='user_id')
    user_last1_activeDay_interact_time['userLast1ActiveDayInteractMinuteRatio'] = (user_last1_activeDay_interact_time['userLast1ActiveDayInteractMinuteNum'] - user_last1_activeDay_interact_time['userDailyInteractMinuteNum'])/ user_last1_activeDay_interact_time['userDailyInteractMinuteNum']
    user_table = pd.merge(user_table,user_last1_activeDay_interact_time[['user_id','userLast1ActiveDayInteractMinuteNum','userLast1ActiveDayInteractMinuteRatio']],how='left',on='user_id')
    user_table = user_table.fillna(0)
    user_table['userLast1ActiveDayInteractMinuteRatio'] = user_table['userLast1ActiveDayInteractMinuteRatio'].apply(lambda x:0 if math.isinf(x) else x)
    #用户过去N天超过平均每活跃日交互时间的天数
    user_time_moreThanDaily = pd.merge(action,action_active_daily_time_num,how='left',on='user_id')
    user_time_moreThanDaily = user_time_moreThanDaily[user_time_moreThanDaily.minuteNum>user_time_moreThanDaily.userDailyInteractMinuteNum]
    user_time_moreThanDaily['userLast%dDayMoreThanDailyInteractTimeDayNum'%N] = 1
    user_time_moreThanDaily = user_time_moreThanDaily[['user_id','userLast%dDayMoreThanDailyInteractTimeDayNum'%N]].groupby(['user_id']).sum().reset_index()
    user_table = pd.merge(user_table,user_time_moreThanDaily[['user_id','userLast%dDayMoreThanDailyInteractTimeDayNum'%N]],how='left',on='user_id')
    user_table = user_table.fillna(0)
    user_table.to_csv(path, index=False, index_label=False)
    return user_table

# 提取用户对 cate=8 的商品的相关特征
def user_Cate8_feature(user_table,action_data,end_date,flag,N=4):
    folder = ['train','test']
    path = "./tmp/%s/user_Cate8_feature%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_table  = pd.read_csv(path)
        return user_table
    print 'user_Cate8_feature'
    
    start_date = datetime.strftime((datetime.strptime(end_date,"%Y-%m-%d")-timedelta(days=N)),"%Y-%m-%d")
    #最后一件交互商品品类是否cate8
    user_isLastCate8 = action_data[
        (action_data.time>=start_date)&(action_data.time<end_date)][['user_id','time','timeHour','timeMinute','cate']].sort(
        ['user_id','time','timeHour','timeMinute'],ascending=False).drop_duplicates(['user_id'])
    user_isLastCate8['userLastInterActIsCate8'] =  user_isLastCate8['cate'].apply(lambda x:1 if x==8 else 0)
    user_table = pd.merge(user_table,user_isLastCate8[['user_id','userLastInterActIsCate8']],how='left',on='user_id')
    user_table = user_table.fillna(0)
    print '最后一件交互商品品类是否cate8'
    #最后一个活跃天Cate8排名
    user_last_active_date = action_data[(action_data.time<end_date)][['user_id','time']].sort(ascending=False).drop_duplicates(['user_id'])
    user_last_active = pd.merge(action_data,user_last_active_date,how='inner',on=['user_id','time'])
    tmp1 = user_last_active[['user_id','sku_id','cate']].drop_duplicates()
    tmp1['productNum']=1
    tmp2 = tmp1.groupby(['user_id','cate']).sum().reset_index()
    tmp2['userLastActiveDayProductCate8Rank'] = tmp2.groupby(['user_id'])['productNum'].rank(method='min',ascending=False)
    userLastActiveDayHasInteractCate8andRank = tmp2[tmp2.cate==8]
    userLastActiveDayHasInteractCate8andRank['userLastActiveDayProductCate8Num']=userLastActiveDayHasInteractCate8andRank['productNum']
    user_table = pd.merge(user_table,userLastActiveDayHasInteractCate8andRank[
            ['user_id','userLastActiveDayProductCate8Num','userLastActiveDayProductCate8Rank']],how='left',on='user_id')
    user_table['userLastActiveDayProductCate8Rank'] = user_table['userLastActiveDayProductCate8Rank'].fillna(12)
    user_table = user_table.fillna(0)
    print '最后一个活跃天Cate8排名'
     #最后一个活跃天Cate8交互时间和与其他Cate对比的排名
    user_last_active_date = action_data[(action_data.time<end_date)][['user_id','time']].sort(ascending=False).drop_duplicates(['user_id'])
    user_last_active = pd.merge(action_data,user_last_active_date,how='inner',on=['user_id','time'])
    tmp3 = user_last_active[['user_id','cate','timeHour','timeMinute']].drop_duplicates()
    tmp3['minuteNum']=1
    tmp3 = tmp3.groupby(['user_id','cate']).sum().reset_index()
    tmp3['userLastActiveDayTimeCate8Rank'] = tmp3.groupby(['user_id'])['minuteNum'].rank(method='min',ascending=False)
    userLastActiveDayHasInteractCate8andRank = tmp3[tmp3.cate==8]
    userLastActiveDayHasInteractCate8andRank['userLastActiveDayTimeCate8Num'] = userLastActiveDayHasInteractCate8andRank['minuteNum']
    user_table = pd.merge(user_table,userLastActiveDayHasInteractCate8andRank[
            ['user_id','userLastActiveDayTimeCate8Num','userLastActiveDayTimeCate8Rank']],how='left',on='user_id')
    user_table['userLastActiveDayTimeCate8Rank'] =user_table['userLastActiveDayTimeCate8Rank'].fillna(12)
    user_table = user_table.fillna(0)
    print '最后一个活跃天Cate8交互时间和与其他Cate对比的排名'
    #最后一个活跃天交互时间最长的商品是否Cate8
    tmp4 = user_last_active[['user_id','sku_id','cate','timeHour','timeMinute']].drop_duplicates()
    tmp4['minuteNum']=1
    tmp4 = tmp4.groupby(['user_id','sku_id','cate']).sum().reset_index()
    tmp4['productRank'] = tmp4.groupby(['user_id'])['minuteNum'].rank(method='min',ascending=False)
    user_cate8 = tmp4[(tmp4.cate==8)].sort(['user_id','productRank']).drop_duplicates(['user_id'])
    user_cate8['userLastAcitveDayCate8ProductMostInteractTime']=user_cate8['minuteNum']
    user_cate8['userLastActiveDayMostProductIsCate8']=user_cate8['productRank'].apply(lambda x:0 if x>1 else 1)
    user_table = pd.merge(user_table,user_cate8[['user_id','userLastAcitveDayCate8ProductMostInteractTime',
                                                 'userLastActiveDayMostProductIsCate8']].drop_duplicates(),how='left',on='user_id')
    user_table = user_table.fillna(0)
    print '最后一个活跃天交互时间最长的商品是否Cate8'
    #过去N天用户有多少天交互Cate8
    user_lastDays = action_data[
        (action_data.time>=start_date)&(action_data.time<end_date)&(action_data.cate==8)][['user_id','time']].drop_duplicates()
    user_lastDays['userLast%dDayInteractCate8Days'%N] = 1
    tmp5 = user_lastDays.groupby(['user_id']).sum().reset_index()
    user_table = pd.merge(user_table,tmp5[['user_id','userLast%dDayInteractCate8Days'%N]],how='left',on='user_id')
    user_table = user_table.fillna(0)
    print '过去N天用户有多少天交互Cate8'
    user_table.to_csv(path, index=False, index_label=False)
    
    return user_table

# 用户在截止日前 最后一个活跃日的交互操作数 和 活跃日日均交互操作数
def user_last1_activeDay_interact_times(user_table,action_data,end_date,flag):
    folder = ['train','test']
    path = "./tmp/%s/user_last1_activeDay_interact_times%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_table  = pd.read_csv(path)
        return user_table
    print 'user_last1_activeDay_interact_times'
    
    #活跃日日均交互操作数(排除掉315的数据和购买当天数据)
    action_active_daily_action_num = action_data[(action_data.time<end_date)&(action_data.time!='2016-03-15')][['user_id','time','userBuy']]
    action_active_daily_action_num['actionNum'] = 1
    action_active_daily_action_num = action_active_daily_action_num[['user_id','time','userBuy','actionNum']].groupby(['user_id','time']).sum().reset_index()
    action_active_daily_action_num['dayNum'] = 1
    action_active_daily_action_num = action_active_daily_action_num[
        action_active_daily_action_num.userBuy==0].groupby(['user_id']).sum().reset_index()
    action_active_daily_action_num['userDailyInteractActionNum'] = action_active_daily_action_num['actionNum']/action_active_daily_action_num['dayNum']
    print '活跃日日均交互操作数'
    #最后一个活跃日交互的商品数
    user_last1_activeDay_interact_action_num = action_data[action_data.time<end_date][['user_id','time']]
    user_last1_activeDay_interact_action_num['userLast1ActiveDayInteractActionNum'] = 1
    user_last1_activeDay_interact_action_num = user_last1_activeDay_interact_action_num.groupby(
        ['user_id','time']).sum().reset_index().sort(['user_id','time'],ascending=False).drop_duplicates(['user_id'])
    print '最后一个活跃日交互的操作数'
    #合并
    user_last1_activeDay_interact_action = pd.merge(
        user_last1_activeDay_interact_action_num[['user_id','userLast1ActiveDayInteractActionNum']],
        action_active_daily_action_num[['user_id','userDailyInteractActionNum']],how='left',on='user_id')
    user_last1_activeDay_interact_action['userLast1ActiveDayInteractActionRatio'] = (user_last1_activeDay_interact_action['userLast1ActiveDayInteractActionNum'] - user_last1_activeDay_interact_action['userDailyInteractActionNum'])/ user_last1_activeDay_interact_action['userDailyInteractActionNum']
    user_table=pd.merge(user_table,user_last1_activeDay_interact_action[['user_id','userDailyInteractActionNum',
                'userLast1ActiveDayInteractActionNum','userLast1ActiveDayInteractActionRatio']],how='left',on='user_id')
    user_table = user_table.fillna(0)
    user_table['userLast1ActiveDayInteractActionRatio'] = user_table['userLast1ActiveDayInteractActionRatio'].apply(lambda x:0 if math.isinf(x) else x)
    user_table.to_csv(path, index=False, index_label=False)
    
    return user_table


# 商品特征
# 创建当前用户总体特征
def product_active_all_statistics(product_table,action_data,end_date,flag):
    
    folder = ['train','test']
    path = "./tmp/%s/product_active_all_statistics%s.csv" % (folder[flag],end_date)
    
    if os.path.exists(path):
        print "There is csv!!!",path
        product_table  = pd.read_csv(path)
        return product_table
    
    start_date = '2016-02-01'
    action_data = action_data[(action_data.time >= start_date) & (action_data.time < end_date)]
    action_data_day = action_data[['sku_id', 'time','cate','userLook','userAdd','userDelete','userBuy','userFavor','userClick']]
    # 操作次数计算
    actionTimes = action_data_day.groupby(['sku_id','cate']).sum().reset_index()
    # 任意操作
    # 【操作总次数】
    actionTimes['productActionNum'] = actionTimes['userLook'] + actionTimes['userAdd'] + actionTimes['userDelete'] +                                    actionTimes['userBuy'] + actionTimes['userFavor'] + actionTimes['userClick']
    # 单一操作
    # 六个【操作特征】转名字
    actionTimes.rename(columns={
        'userLook': 'productLook', 'userAdd': 'productAdd', 'userDelete': 'productDelete',
        'userBuy': 'productBuy', 'userFavor': 'productFavor', 'userClick': 'productClick'},
        inplace=True)
    # 【各操作/操作比】
    actionTimes['productLookActionRatio'] = actionTimes['productLook'] / actionTimes['productActionNum']
    actionTimes['productAdd-ActionRatio'] = actionTimes['productAdd'] / actionTimes['productActionNum']
    actionTimes['productDeleteActionRatio'] = actionTimes['productDelete'] / actionTimes['productActionNum']
    actionTimes['productBuyActionRatio'] = actionTimes['productBuy'] / actionTimes['productActionNum']
    actionTimes['productFavorActionRatio'] = actionTimes['productFavor'] / actionTimes['productActionNum']
    actionTimes['productClickActionRatio'] = actionTimes['productClick'] / actionTimes['productActionNum']
    # 【购买/加购比】【购买/收藏比】【购买/点击比】【购买/浏览比】
    actionTimes['productBuyAddRatio'] = actionTimes['productBuy'] / actionTimes['productAdd']
    actionTimes['productBuyAddRatio'] = actionTimes['productBuyAddRatio'].apply(lambda x:100 if (math.isinf(x)) else x)
    actionTimes['productBuyFavorRatio'] = actionTimes['productBuy'] / actionTimes['productFavor']
    actionTimes['productBuyFavorRatio'] = actionTimes['productBuyFavorRatio'].apply(lambda x:100 if (math.isinf(x)) else x)   
    actionTimes['productBuyClickRatio'] = actionTimes['productBuy'] / actionTimes['productClick']
    actionTimes['productBuyClickRatio'] = actionTimes['productBuyClickRatio'].apply(lambda x:100 if (math.isinf(x)) else x) 
    actionTimes['productBuyLookRatio'] = actionTimes['productBuy'] / actionTimes['productLook']
    actionTimes['productBuyLookRatio'] = actionTimes['productBuyLookRatio'].apply(lambda x:100 if (math.isinf(x)) else x) 
    # 【加购/收藏比】【加购/点击比】【加购/浏览比】
    actionTimes['productAddFavorRatio'] = actionTimes['productAdd'] / actionTimes['productFavor']
    actionTimes['productAddFavorRatio'] = actionTimes['productAddFavorRatio'].apply(lambda x:100 if (math.isinf(x)) else x) 
    actionTimes['productAddClickRatio'] = actionTimes['productAdd'] / actionTimes['productClick']
    actionTimes['productAddClickRatio'] = actionTimes['productAddClickRatio'].apply(lambda x:100 if (math.isinf(x)) else x) 
    actionTimes['productAddLookRatio'] = actionTimes['productAdd'] / actionTimes['productLook']
    actionTimes['productAddLookRatio'] = actionTimes['productAddLookRatio'].apply(lambda x:100 if (math.isinf(x)) else x) 
    # 【收藏/点击比】【收藏/浏览比】
    actionTimes['productFavorClickRatio'] = actionTimes['productFavor'] / actionTimes['productClick']
    actionTimes['productFavorClickRatio'] = actionTimes['productFavorClickRatio'].apply(lambda x:100 if (math.isinf(x)) else x)
    actionTimes['productFavorLookRatio'] =  actionTimes['productFavor'] / actionTimes['productLook']
    actionTimes['productFavorLookRatio'] =  actionTimes['productFavorLookRatio'].apply(lambda x:100 if (math.isinf(x)) else x)
    # 【点击/浏览比】
    actionTimes['productClickLookRatio'] =  actionTimes['productClick'] / actionTimes['productLook']
    actionTimes['productClickLookRatio'] =  actionTimes['productClickLookRatio'].apply(lambda x:100 if (math.isinf(x)) else x)
    # 【删购/加购比】
    actionTimes['productDeleteAddRatio'] =  actionTimes['productDelete'] / actionTimes['productAdd']
    actionTimes['productDeleteAddRatio'] =  actionTimes['productDeleteAddRatio'].apply(lambda x:100 if (math.isinf(x)) else x)
    actionTimes = actionTimes.drop(['cate'],1)
    product_table = pd.merge(product_table,actionTimes,how='left',on='sku_id')
    #缺失值填充，排名的缺失值应该填排名的最大值+1
    product_table = product_table.fillna(0)
    
    # 【被交互的总人数】
    action_user_interactive = (action_data[['user_id','sku_id','cate']]).drop_duplicates()
    action_user_interactive['productInteractiveUserNum'] = 1
    action_user_interactive = action_user_interactive[
        ['sku_id','cate','productInteractiveUserNum']].groupby(['sku_id','cate']).sum().reset_index()
    action_user_interactive = action_user_interactive.drop(['cate'],1)
    product_table = pd.merge(product_table, action_user_interactive, how='left', on='sku_id')
    product_table = product_table.fillna(0)
    # 【被购买的总人数】
    action_user_buy = ((action_data[action_data.type==4])[['user_id', 'sku_id','cate']]).drop_duplicates()
    action_user_buy['productBuyUserNum'] = 1
    action_user_buy = action_user_buy[
        ['sku_id','cate','productBuyUserNum']].groupby(['sku_id','cate']).sum().reset_index()
    action_user_buy = action_user_buy.drop(['cate'],1)
    product_table = pd.merge(product_table, action_user_buy, how='left', on='sku_id')
    product_table = product_table.fillna(0)
    
    product_table.to_csv(path, index=False, index_label=False)
    return product_table

#创建当前product对单个用户的统计特征(粒度是用户,不是具体某一确定用户) 
def product_last_days_statistics(product_table,action_data,list,end_date,flag):
    
    folder = ['train','test']
    path = "./tmp/%s/product_last_days_statistics%s.csv" % (folder[flag],end_date)
    
    if os.path.exists(path):
        print "There is csv!!!",path
        product_table  = pd.read_csv(path)
        return product_table
    
    start_date = '2016-02-01'
    # 商品前30日日均6类交互行为的数量
    start_date = datetime.strftime((datetime.strptime(end_date,"%Y-%m-%d")-timedelta(days=30)),"%Y-%m-%d")
    print "start_date=",start_date
    actionNum_pDay = (action_data[(action_data.time >= start_date) & (action_data.time < end_date)])[
        ['sku_id', 'time','userLook','userAdd','userDelete','userBuy','userFavor','userClick']]
    daysNum = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    
    actionNum_pDay.rename(columns={
        'userLook':'productDailyActionType_1','userAdd':'productDailyActionType_2','userDelete':'productDailyActionType_3',
        'userBuy':'productDailyActionType_4','userFavor':'productDailyActionType_5','userClick':'productDailyActionType_6'},inplace = True)
    
    actionNum_pDay = actionNum_pDay.groupby(['sku_id']).sum().reset_index()
    actionNum_pDay['productDailyActionType_1'] = actionNum_pDay['productDailyActionType_1'] / daysNum
    actionNum_pDay['productDailyActionType_2'] = actionNum_pDay['productDailyActionType_2'] / daysNum
    actionNum_pDay['productDailyActionType_3'] = actionNum_pDay['productDailyActionType_3'] / daysNum
    actionNum_pDay['productDailyActionType_4'] = actionNum_pDay['productDailyActionType_4'] / daysNum
    actionNum_pDay['productDailyActionType_5'] = actionNum_pDay['productDailyActionType_5'] / daysNum
    actionNum_pDay['productDailyActionType_6'] = actionNum_pDay['productDailyActionType_6'] / daysNum
    product_table = pd.merge(product_table, actionNum_pDay, how='left', on='sku_id')
    product_table = product_table.fillna(0)
    for i in list:
        print "在购买的前 %d 天特征\n" % (i,)
        start_date = (datetime.strptime(
            end_date,"%Y-%m-%d")-timedelta(days=i)).strftime("%Y-%m-%d")
        action_data = action_data[(action_data.time>=start_date)&(action_data.time<=end_date)]
        #商品6种交互行为数量
        product_action = action_data[['sku_id','type']]
        type_dummy = pd.get_dummies(product_action['type'], prefix="productLast%dDayNumtype"%i)
        product_action = pd.concat([product_action[['sku_id']], type_dummy], axis=1)
        product_action = product_action.groupby(['sku_id']).sum().reset_index()
        # 商品6种交互行为数量和其日均交互行为数量的比值
        user_action = pd.merge(product_action, actionNum_pDay, how='left', on='sku_id')
        user_action["productLast%dDayNumtype_1Ratio"% i] = \
                   user_action["productLast%dDayNumtype_1" % i] / user_action['productDailyActionType_1'] / i
        user_action["productLast%dDayNumtype_2Ratio"% i] =  \
                   user_action["productLast%dDayNumtype_2" % i] / user_action['productDailyActionType_2'] / i
        user_action["productLast%dDayNumtype_3Ratio" % i] =  \
                   user_action["productLast%dDayNumtype_3" % i] / user_action['productDailyActionType_3'] / i
        user_action["productLast%dDayNumtype_4Ratio" % i] = \
                   user_action["productLast%dDayNumtype_4" % i] / user_action['productDailyActionType_4'] / i
        user_action["productLast%dDayNumtype_5Ratio" % i] = \
                   user_action["productLast%dDayNumtype_5" % i] / user_action['productDailyActionType_5'] / i
        user_action["productLast%dDayNumtype_6Ratio" % i] = \
                   user_action["productLast%dDayNumtype_6" % i] / user_action['productDailyActionType_6'] / i
        user_action = user_action.drop(['productDailyActionType_1', 'productDailyActionType_2', 'productDailyActionType_3',
                                        'productDailyActionType_4', 'productDailyActionType_5', 'productDailyActionType_6'],1)
        
        product_table = pd.merge(product_table, user_action, how='left', on='sku_id')
        # 【商品交互过的用户数】
        user = action_data[['user_id', 'sku_id']]
        user_interactive = user.drop_duplicates()
        user_interactive['productLast%dDaysInteractiveUsers'%i] = 1
        user_interactive = user_interactive[
            ['sku_id', 'productLast%dDaysInteractiveUsers'%i]].groupby(['sku_id']).sum().reset_index()
        product_table = pd.merge(product_table, user_interactive, how='left', on='sku_id')
        product_table = product_table.fillna(0)
        
        product_table.to_csv(path, index=False, index_label=False)

    return product_table


#创建当前product对单个用户的统计特征(粒度是用户,不是具体某一确定用户)
def product_active_statistic_user(product_table,action_data,end_date,flag):
    
    folder = ['train','test']
    path = "./tmp/%s/product_active_statistic_user%s.csv" % (folder[flag],end_date)

    if os.path.exists(path):
        print "There is csv!!!",path
        product_table  = pd.read_csv(path)
        return product_table
    
    start_date = '2016-02-01'
    action_data = action_data[
        (action_data.time >= start_date) & (action_data.time < end_date)]

    # 商品对交互过得用户的平均交互数，最大交互数，最小交互数
    product_interactive = action_data[['user_id', 'sku_id']]
    product_interactive['activeNum'] = 1
    product_interactive = product_interactive.groupby(
        ['user_id', 'sku_id']).sum().reset_index()
    product_interactive['activeNum'] = product_interactive['activeNum'].astype('str')
    product_interactive = product_interactive[['sku_id','activeNum']].groupby(
        ['sku_id'])['activeNum'].agg(lambda x: ':'.join(x)).reset_index()
    product_interactive['productMeanInteractive'] = product_interactive.activeNum.apply(
        lambda x: np.mean([int(d) for d in x.split(':')]))
    product_interactive['productMinInteractive'] = product_interactive.activeNum.apply(
        lambda x: min([int(d) for d in x.split(':')]))
    product_interactive['productMaxInteractive'] = product_interactive.activeNum.apply(
        lambda x: max([int(d) for d in x.split(':')]))
    product_interactive = product_interactive.drop(['activeNum'], 1)
    product_table = pd.merge(product_table, product_interactive, how='left', on='sku_id')
    product_table = product_table.fillna(0)
    # 商品对交互过得用户的平均购买数，最大购买数
    product_buy = (action_data[action_data.type == 4])[['user_id', 'sku_id']]
    product_buy['buyNum'] = 1
    product_buy = product_buy.groupby(
        ['user_id', 'sku_id']).sum().reset_index()
    product_buy['buyNum'] = product_buy['buyNum'].astype('str')
    product_buy = product_buy[['sku_id','buyNum']].groupby(
        ['sku_id'])['buyNum'].agg(lambda x: ':'.join(x)).reset_index()
    product_buy['productMeanBuy'] = product_buy.buyNum.apply(
        lambda x: np.mean([int(d) for d in x.split(':')]))
    product_buy['productMaxBuy'] = product_buy.buyNum.apply(
        lambda x: max([int(d) for d in x.split(':')]))
    product_buy = product_buy.drop(['buyNum'], 1)
    product_table = pd.merge(product_table, product_buy, how='left', on='sku_id')
    product_table = product_table.fillna(0)
    
    # 商品被购买过的用户数量
    product_buy_num = (action_data[action_data.type == 4])[['user_id', 'sku_id']]
    product_buy_num = product_buy_num.drop_duplicates()
    product_buy_num['productBuyUserNum'] = 1
    product_buy_num = product_buy_num[['sku_id', 'productBuyUserNum']].groupby(
        ['sku_id']).sum().reset_index()
    product_table = pd.merge(product_table, product_buy_num, how='left', on='sku_id')
    product_table = product_table.fillna(0)
    
    # 商品被重复购买的次数，回购的用户总数，回购率
    product_buyrepeat = (action_data[action_data.type == 4])[['user_id', 'sku_id']]
    product_buyrepeat['buyNum'] = 1
    product_buyrepeat = product_buyrepeat.groupby(
        ['user_id', 'sku_id']).sum().reset_index()
    # 回购的用户总数
    product_buyrepeatnum = product_buyrepeat[product_buyrepeat.buyNum > 1]
    product_buyrepeatnum['productBuyUserRepeat'] = 1
    product_buyrepeatnum = product_buyrepeatnum[
        ['sku_id', 'productBuyUserRepeat']].groupby(['sku_id']).sum().reset_index()
    product_table = pd.merge(product_table, product_buyrepeatnum, how='left', on='sku_id')
    # 重复购买的次数
    product_buyrepeat['productBuyTimeRepeat'] = map(
        lambda x: x - 1, product_buyrepeat['buyNum'])
    product_buyrepeattime = product_buyrepeat[['sku_id', 'productBuyTimeRepeat']].groupby(
        ['sku_id']).sum().reset_index()
    product_table = pd.merge(product_table, product_buyrepeattime, how='left', on='sku_id')
    # 重复购买率
    product_buyrepeatratio = product_buyrepeatnum[['sku_id','productBuyUserRepeat']]
    product_buyrepeatratio = pd.merge(product_buyrepeatratio,product_buy_num[['sku_id','productBuyUserNum']],how='left',on='sku_id')
    product_buyrepeatratio['productBuyUserRepeatRatio'] =         product_buyrepeatratio['productBuyUserRepeat']/product_buyrepeatratio['productBuyUserNum']
    product_buyrepeatratio = product_buyrepeatratio.drop(['productBuyUserRepeat','productBuyUserNum'],1)
    product_table = pd.merge(product_table, product_buyrepeatratio, how='left', on='sku_id')
    product_table = product_table.fillna(0)
    
    product_table.to_csv(path, index=False, index_label=False)
    
    return product_table


# 用户-商品 交互特征
#预测期前N天该用户对该商品的6种行为数
def get_last_days_action(user_product_table,action_data,last_time_list,end_date,flag):
    
    folder = ['train','test']
    path = "./tmp/%s/get_last_days_action%s.csv" % (folder[flag],end_date)
    
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    for i in last_time_list:
        print "在购买的前 %d 天特征\n" % (i,)
        start_date = (datetime.strptime(
            end_date,"%Y-%m-%d")-timedelta(days=i)).strftime("%Y-%m-%d")
        action_data = action_data[(action_data.time>=start_date)&(action_data.time<end_date)]
        #用户-商品6种交互行为数量
        action = action_data[['user_id','sku_id','type']]
        type_dummy = pd.get_dummies(action['type'], prefix="upLast%dDayNumtype"%i)
        action = pd.concat([action[['user_id','sku_id']], type_dummy], axis=1)
        action = action.groupby(
        ['user_id','sku_id']).sum().reset_index()
        user_product_table = pd.merge(user_product_table, action, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
    
    user_product_table.to_csv(path, index=False, index_label=False)
    return user_product_table

# 预测期前N天该用户对该商品的6种行为数的排序以及总行为次数的排序
def get_last_days_action_rank(user_product_table,action_data,last_time_list,end_date,flag):
    folder = ['train','test']
    path = "./tmp/%s/get_last_days_action_rank%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    print 'get_last_days_action_rank'
    for i in last_time_list:
        print "在购买的前 %d 天特征\n" % (i,)
        start_date = (datetime.strptime(
            end_date,"%Y-%m-%d")-timedelta(days=i)).strftime("%Y-%m-%d")
        action_data = action_data[(action_data.time>=start_date)&(action_data.time<end_date)]
        # 用户-商品6种交互行为数量
        action = action_data[['user_id','sku_id','type']]
        type_dummy = pd.get_dummies(action['type'], prefix="upLast%dDayNumtype"%i)
        action = pd.concat([action[['user_id','sku_id']], type_dummy], axis=1)
        action = action.groupby(
        ['user_id','sku_id']).sum().reset_index()
        
        # 统计用户对该商品的交互行为的总数量
        action['upLast%dDayNumUserProductInteractive'%i] = 0
        # 6种行为的排序  
        for t in range(1,7):
            tmp_record = action[['user_id','sku_id','upLast%dDayNumtype_%d'%(i,t)]]
            tmp_record['upLast%dDayNumtype_%d_rank'%(i,t)] = tmp_record.groupby('user_id')['upLast%dDayNumtype_%d'%(i,t)].rank(
                ascending=False,method='max')

            action = pd.merge(action, tmp_record, how='left',on=['user_id','sku_id','upLast%dDayNumtype_%d'%(i,t)])
            action['upLast%dDayNumUserProductInteractive'%i] += action['upLast%dDayNumtype_%d'%(i,t)]
        # 总交互行为次数的排序
        tmp_record = action[['user_id','sku_id','upLast%dDayNumUserProductInteractive'%i]]
        tmp_record['upLast%dDayNumUserProductInteractive_rank'%i] = \
            tmp_record.groupby('user_id')['upLast%dDayNumUserProductInteractive'%i].rank(ascending=False,method='max')
        action = pd.merge(action, tmp_record, how='left',on=['user_id','sku_id','upLast%dDayNumUserProductInteractive'%i])
        
        for t in range(1,7):
            del action['upLast%dDayNumtype_%d'%(i,t)]
        del action['upLast%dDayNumUserProductInteractive'%i]
        
        user_product_table = pd.merge(user_product_table, action, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
    user_product_table.to_csv(path, index=False, index_label=False)
    return user_product_table

# 交互行为的时间特征
def action_get_interactive_days(user_product_table,action_data,end_date,flag):
    start_date = '2016-02-01'
    
    folder = ['train','test']
    path = "./tmp/%s/action_get_interactive_days%s.csv" % (folder[flag],end_date)
    
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    print 'action_get_interactive_days' 
    action_data = action_data[(action_data.time>=start_date)&(action_data.time<end_date)]
    interval = datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    interval = interval.total_seconds()/(24*60*60)+1
   
    record = action_data[['user_id','sku_id','type','time']]
    # 升序排序
    last_interactive = record.sort(['user_id','sku_id','time'],ascending=True)
    print '所有交互行为统计'
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = last_interactive.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 用户对商品的最晚交互时间到预测期的时间
    time['user_product_LastActiveDay'] = time['time'].apply(
        lambda x: (x.split(':'))[len(x.split(':'))-1])
    time['user_product_LastActiveDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_LastActiveDay'])
    time['intervalFromLastActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_LastActiveDay'])
    # 用户对商品的交互天数
    time['user_product_firstActiveDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstActiveDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstActiveDay'])
    # 计算间隔
    time['user_product_firstLastDayInterval'] = time['user_product_LastActiveDay'] - time['user_product_firstActiveDay']
    time['user_product_firstLastDayInterval'] = map(
        lambda x:x/np.timedelta64(1,'D')+1,time['user_product_firstLastDayInterval'])

    del time['user_product_LastActiveDay']
    del time['user_product_firstActiveDay']
    del time['time']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    
    # 购买行为
    print '购买行为特征统计'
    time = last_interactive[last_interactive.type==4]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早购买该商品的时间
    time['user_product_firstBuyDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstBuyDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstBuyDay'])
    # 最晚购买该商品的时间
    time['user_product_lastBuyDay'] = time['time'].apply(
        lambda x: (x.split(':'))[len(x.split(':'))-1])
    time['user_product_lastBuyDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_lastBuyDay'])
    # 用户购买该商品的时间间隔
    time['user_product_firstLastDayBuyInterval'] = time['user_product_lastBuyDay'] - time['user_product_firstBuyDay']
    time['user_product_firstLastDayBuyInterval'] = map(
        lambda x:x/np.timedelta64(1,'D')+1,time['user_product_firstLastDayBuyInterval'])
    # 用户购买该商品的最晚时间距离预测期的时间
    time['intervalBuyFromLastActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_lastBuyDay'])
    del time['time']
    del time['user_product_lastBuyDay']
    del time['user_product_firstBuyDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    # 加入购物车行为
    print '加入购物车行为特征统计'
    time = last_interactive[last_interactive.type==2]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早将该商品加入购物车的时间
    time['user_product_firstAddCartDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstAddCartDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstAddCartDay'])
    # 最晚将该商品加入购物车的时间
    time['user_product_lastAddCartDay'] = time['time'].apply(
        lambda x: (x.split(':'))[len(x.split(':'))-1])
    time['user_product_lastAddCartDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_lastAddCartDay'])
    # 用户将该商品加入购物车的时间间隔
    time['user_product_firstLastDayAddCartInterval'] = time['user_product_lastAddCartDay'] - time['user_product_firstAddCartDay']
    time['user_product_firstLastDayAddCartInterval'] = map(
        lambda x:x/np.timedelta64(1,'D')+1,time['user_product_firstLastDayAddCartInterval'])
    # 用户将该商品加入购物车的最晚时间距离预测期的时间
    time['intervalAddCartFromLastActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_lastAddCartDay'])
    del time['time']
    del time['user_product_lastAddCartDay']
    del time['user_product_firstAddCartDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    # 关注行为
    print '关注行为统计'
    time = last_interactive[last_interactive.type==5]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早关注该商品的时间
    time['user_product_firstFavorDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstFavorDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstFavorDay'])
    # 最晚关注该商品的时间
    time['user_product_lastFavorDay'] = time['time'].apply(
        lambda x: (x.split(':'))[len(x.split(':'))-1])
    time['user_product_lastFavorDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_lastFavorDay'])
    # 用户关注该商品的时间间隔
    time['user_product_firstLastDayFavorInterval'] = time['user_product_lastFavorDay'] - time['user_product_firstFavorDay']
    time['user_product_firstLastDayFavorInterval'] = map(
        lambda x:x/np.timedelta64(1,'D')+1,time['user_product_firstLastDayFavorInterval'])
    # 用户关注该商品的最晚时间距离预测期的时间
    time['intervalFavorFromLastActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_lastFavorDay'])
    del time['time']
    del time['user_product_lastFavorDay']
    del time['user_product_firstFavorDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    # 浏览行为
    print '浏览行为统计'
    time = last_interactive[last_interactive.type==1]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早浏览该商品的时间
    time['user_product_firstLookDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstLookDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstLookDay'])
    # 最晚浏览该商品的时间
    time['user_product_lastLookDay'] = time['time'].apply(
        lambda x: (x.split(':'))[len(x.split(':'))-1])
    time['user_product_lastLookDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_lastLookDay'])
    # 用户浏览该商品的时间间隔
    time['user_product_firstLastDayLookInterval'] = time['user_product_lastLookDay'] - time['user_product_firstLookDay']
    time['user_product_firstLastDayLookInterval'] = map(
        lambda x:x/np.timedelta64(1,'D')+1,time['user_product_firstLastDayLookInterval'])
    # 用户浏览该商品的最晚时间距离预测期的时间
    time['intervalLookFromLastActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_lastLookDay'])
    del time['time']
    del time['user_product_lastLookDay']
    del time['user_product_firstLookDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    # 点击行为
    print '点击行为统计'
    time = last_interactive[last_interactive.type==6]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早点击该商品的时间
    time['user_product_firstClickDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstClickDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstClickDay'])
    # 最晚点击该商品的时间
    time['user_product_lastClickDay'] = time['time'].apply(
        lambda x: (x.split(':'))[len(x.split(':'))-1])
    time['user_product_lastClickDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_lastClickDay'])
    # 用户点击该商品的时间间隔
    time['user_product_firstLastDayClickInterval'] = time['user_product_lastClickDay'] - time['user_product_firstClickDay']
    time['user_product_firstLastDayClickInterval'] = map(
        lambda x:x/np.timedelta64(1,'D')+1,time['user_product_firstLastDayClickInterval'])
    # 用户点击该商品的最晚时间距离预测期的时间
    time['intervalClickFromLastActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_lastClickDay'])
    del time['time']
    del time['user_product_lastClickDay']
    del time['user_product_firstClickDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    user_product_table.to_csv(path, index=False, index_label=False)
    
    return user_product_table


# 交互行为的时间特征 生成行为第一次交互时间到预测期的间隔
def action_get_interactive_days_2(user_product_table,action_data,end_date,flag):
    start_date = '2016-02-01'
    folder = ['train','test']
    path = "./tmp/%s/action_get_interactive_days_2%s.csv" % (folder[flag],end_date)
    
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    
    print 'action_get_interactive_days_2'
    action_data = action_data[(action_data.time>=start_date)&(action_data.time<end_date)]
    interval = datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    interval = interval.total_seconds()/(24*60*60)+1
   
    record = action_data[['user_id','sku_id','type','time']]
    # 升序排序
    last_interactive = record.sort_values(by=['user_id','sku_id','time'],ascending=True)

    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = last_interactive.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    
    time['user_product_firstActiveDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstActiveDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstActiveDay'])
    # 用户对商品的最早交互时间到预测期的时间
    time['intervalFromFirstActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_firstActiveDay'])
    
    del time['user_product_firstActiveDay']
    del time['time']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    
    # 购买行为
    time = last_interactive[last_interactive.type==4]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早购买该商品的时间
    time['user_product_firstBuyDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstBuyDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstBuyDay'])
    
    # 用户购买该商品的最早时间距离预测期的时间
    time['intervalBuyFromFirstActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_firstBuyDay'])
    del time['time']
    del time['user_product_firstBuyDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    # 加入购物车行为
    time = last_interactive[last_interactive.type==2]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早将该商品加入购物车的时间
    time['user_product_firstAddCartDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstAddCartDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstAddCartDay'])
    # 用户将该商品加入购物车的最早时间距离预测期的时间
    time['intervalAddCartFromFirstActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_firstAddCartDay'])
    del time['time']
    del time['user_product_firstAddCartDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    # 关注行为
    time = last_interactive[last_interactive.type==5]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早关注该商品的时间
    time['user_product_firstFavorDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstFavorDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstFavorDay'])
    # 用户关注该商品的最早时间距离预测期的时间
    time['intervalFavorFromFirstActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_firstFavorDay'])
    del time['time']
    del time['user_product_firstFavorDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    # 浏览行为
    time = last_interactive[last_interactive.type==1]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早浏览该商品的时间
    time['user_product_firstLookDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstLookDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstLookDay'])
    # 用户浏览该商品的最早时间距离预测期的时间
    time['intervalLookFromFirstActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_firstLookDay'])
    del time['time']
    del time['user_product_firstLookDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    # 点击行为
    time = last_interactive[last_interactive.type==6]
    #用户的操作时间连接成一行，time只保留user_id,sku_id和time字段
    time = time.groupby(['user_id','sku_id'])['time'].agg(
        lambda x:':'.join(x)).reset_index()
    # 最早点击该商品的时间
    time['user_product_firstClickDay'] = time['time'].apply(
        lambda x: (x.split(':'))[0])
    time['user_product_firstClickDay'] = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), time['user_product_firstClickDay'])
    # 用户点击该商品的最早时间距离预测期的时间
    time['intervalClickFromFirstActiveDayToPredict'] = map(
        lambda x: (datetime.strptime(end_date, "%Y-%m-%d")-x).days, time['user_product_firstClickDay'])
    del time['time']
    del time['user_product_firstClickDay']
    user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    user_product_table = user_product_table.fillna(interval)
    
    user_product_table.to_csv(path, index=False, index_label=False)
    
    return user_product_table

# 交互行为的时间特征的排序, 输入的user_product_table应该是经过提取了交互行为时间特征的结果,总共增加18个特征(5个行为+1个总交互)
def action_get_interactive_days_rank(user_product_table,end_date,flag):
    start_date = '2016-02-01'
    folder = ['train','test']
    path = "./tmp/%s/action_get_interactive_days_rank%s.csv" % (folder[flag],end_date)
    
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    print 'action_get_interactive_days_rank'
    interval = datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    interval = interval.total_seconds()/(24*60*60)+1
   
    # 用户对商品的最晚交互时间到预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalFromLastActiveDayToPredict']]
    time['intervalFromLastActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalFromLastActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalFromLastActiveDayToPredict']
    del user_product_table['intervalFromLastActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
        
    # 用户对商品的交互天数的排序
    time = user_product_table[['user_id','sku_id','user_product_firstLastDayInterval']]
    time['user_product_firstLastDayInterval_rank'] = \
        time.groupby('user_id')['user_product_firstLastDayInterval'].rank(ascending=False,method='max')
    del time['user_product_firstLastDayInterval']
    del user_product_table['user_product_firstLastDayInterval']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
        
    # 用户对商品的最早交互时间到预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalFromFirstActiveDayToPredict']]
    time['intervalFromFirstActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalFromFirstActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalFromFirstActiveDayToPredict']
    del user_product_table['intervalFromFirstActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    
    # 购买行为
    # 用户购买该商品的时间间隔的排序
    time = user_product_table[['user_id','sku_id','user_product_firstLastDayBuyInterval']]
    time['user_product_firstLastDayBuyInterval_rank'] = \
        time.groupby('user_id')['user_product_firstLastDayBuyInterval'].rank(ascending=False,method='max')
    del time['user_product_firstLastDayBuyInterval']
    del user_product_table['user_product_firstLastDayBuyInterval']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户购买该商品的最晚时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalBuyFromLastActiveDayToPredict']]
    time['intervalBuyFromLastActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalBuyFromLastActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalBuyFromLastActiveDayToPredict']
    del user_product_table['intervalBuyFromLastActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户购买该商品的最早时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalBuyFromFirstActiveDayToPredict']]
    time['intervalBuyFromFirstActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalBuyFromFirstActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalBuyFromFirstActiveDayToPredict']
    del user_product_table['intervalBuyFromFirstActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    
    # 加入购物车行为
    # 用户将该商品加入购物车的时间间隔的排序
    time = user_product_table[['user_id','sku_id','user_product_firstLastDayAddCartInterval']]
    time['user_product_firstLastDayAddCartInterval_rank'] = \
        time.groupby('user_id')['user_product_firstLastDayAddCartInterval'].rank(ascending=False,method='max')
    del time['user_product_firstLastDayAddCartInterval']
    del user_product_table['user_product_firstLastDayAddCartInterval']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户将该商品加入购物车的最晚时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalAddCartFromLastActiveDayToPredict']]
    time['intervalAddCartFromLastActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalAddCartFromLastActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalAddCartFromLastActiveDayToPredict']
    del user_product_table['intervalAddCartFromLastActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户将该商品加入购物车的最早时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalAddCartFromFirstActiveDayToPredict']]
    time['intervalAddCartFromFirstActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalAddCartFromFirstActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalAddCartFromFirstActiveDayToPredict']
    del user_product_table['intervalAddCartFromFirstActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    
    # 关注行为
    # 用户关注该商品的时间间隔的排序
    time = user_product_table[['user_id','sku_id','user_product_firstLastDayFavorInterval']]
    time['user_product_firstLastDayFavorInterval_rank'] = \
        time.groupby('user_id')['user_product_firstLastDayFavorInterval'].rank(ascending=False,method='max')
    del time['user_product_firstLastDayFavorInterval']
    del user_product_table['user_product_firstLastDayFavorInterval']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户关注该商品的最晚时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalFavorFromLastActiveDayToPredict']]
    time['intervalFavorFromLastActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalFavorFromLastActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalFavorFromLastActiveDayToPredict']
    del user_product_table['intervalFavorFromLastActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户关注该商品的最早时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalFavorFromFirstActiveDayToPredict']]
    time['intervalFavorFromFirstActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalFavorFromFirstActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalFavorFromFirstActiveDayToPredict']
    del user_product_table['intervalFavorFromFirstActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    
    # 浏览行为
    # 用户浏览该商品的时间间隔的排序
    time = user_product_table[['user_id','sku_id','user_product_firstLastDayLookInterval']]
    time['user_product_firstLastDayLookInterval_rank'] = \
        time.groupby('user_id')['user_product_firstLastDayLookInterval'].rank(ascending=False,method='max')
    del time['user_product_firstLastDayLookInterval']
    del user_product_table['user_product_firstLastDayLookInterval']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户浏览该商品的最晚时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalLookFromLastActiveDayToPredict']]
    time['intervalLookFromLastActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalLookFromLastActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalLookFromLastActiveDayToPredict']
    del user_product_table['intervalLookFromLastActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户浏览该商品的最早时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalLookFromFirstActiveDayToPredict']]
    time['intervalLookFromFirstActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalLookFromFirstActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalLookFromFirstActiveDayToPredict']
    del user_product_table['intervalLookFromFirstActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    
    # 点击行为
    # 用户点击该商品的时间间隔的排序
    time = user_product_table[['user_id','sku_id','user_product_firstLastDayClickInterval']]
    time['user_product_firstLastDayClickInterval_rank'] = \
        time.groupby('user_id')['user_product_firstLastDayClickInterval'].rank(ascending=False,method='max')
    del time['user_product_firstLastDayClickInterval']
    del user_product_table['user_product_firstLastDayClickInterval']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户点击该商品的最晚时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalClickFromLastActiveDayToPredict']]
    time['intervalClickFromLastActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalClickFromLastActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalClickFromLastActiveDayToPredict']
    del user_product_table['intervalClickFromLastActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    # 用户点击该商品的最早时间距离预测期的时间的排序
    time = user_product_table[['user_id','sku_id','intervalClickFromFirstActiveDayToPredict']]
    time['intervalClickFromFirstActiveDayToPredict_rank'] = \
        time.groupby('user_id')['intervalClickFromFirstActiveDayToPredict'].rank(ascending=False,method='max')
    del time['intervalClickFromFirstActiveDayToPredict']
    del user_product_table['intervalClickFromFirstActiveDayToPredict']
    user_product_table = pd.merge(
        user_product_table, time, how='left',on=['user_id','sku_id'])
    
    user_product_table.to_csv(path, index=False, index_label=False)
    
    return user_product_table

# 比值类特征,操作次数的比值，增加了 12 个特征, 缺失值填充为0
def action_user_product_NumType_ratio(user_product_table,action_data,last_time_list,end_date,flag):
    
    folder = ['train','test']
    path = "./tmp/%s/action_user_product_NumType_ratio%s.csv" % (folder[flag],end_date)
    
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    
    print 'action_user_product_NumType_ratio'
    for i in last_time_list:
        print "在购买的前 %d 天特征\n" % (i,)
        start_date = (datetime.strptime(
            end_date,"%Y-%m-%d")-timedelta(days=i)).strftime("%Y-%m-%d")
        action_data = action_data[(action_data.time>=start_date)&(action_data.time<end_date)]
######### 预测期前N天内对该商品的操作（购买、收藏、点击等行为）次数/访问次数，6种行为都包括,以及对该商品的所有交互操作总次数 #######
        action_original = action_data[['user_id','sku_id','type']]
        type_dummy = pd.get_dummies(action_original['type'], prefix="upLast%dDayNumtype"%i)
        action_original = pd.concat([action_original[['user_id','sku_id']], type_dummy], axis=1)
        # 统计得到用户对商品的6种交互行为数量
        action = action_original.groupby(
        ['user_id','sku_id']).sum().reset_index()
        
        # 统计用户对该商品的交互行为的总数量
        action['upLast%dDayNumUserProductInteractive'%i] = 0
        for t in range(1,7):
            action['upLast%dDayNumUserProductInteractive'%i] += action['upLast%dDayNumtype_%d'%(i,t)]
        # 统计比值
        for t in range(1,7):
            action['upLast%dDayNumtype_%d_UserProductInteractiveRatio'%(i,t)] = \
                   action['upLast%dDayNumtype_%d'%(i,t)] / action['upLast%dDayNumUserProductInteractive'%i]
                                                    
######### 预测期前N天内对该商品的操作（购买、收藏、点击等行为）次数/用户该操作总次数，6种行为都包括 ############################       
        action_tmp = action_original.drop('sku_id',axis=1)
        # 统计用户6种行为的总操作次数
        action_tmp = action_tmp.groupby(['user_id']).sum().reset_index()
        # 更改名字
        for t in range(1,7):
            action_tmp['upLast%dDayNumtype_%d_userAllProduct'%(i,t)]= action_tmp['upLast%dDayNumtype_%d'%(i,t)]
        for t in range(1,7):
            del action_tmp['upLast%dDayNumtype_%d'%(i,t)]
        
        action = pd.merge(action, action_tmp, how='left', on='user_id')
        # 统计比值,inf数值填充为0
        for t in range(1,7):
            action['upLast%dDayNumtype_%d_userAllProductRatio'%(i,t)] = \
                   action['upLast%dDayNumtype_%d'%(i,t)] / action['upLast%dDayNumtype_%d_userAllProduct'%(i,t)]
            action['upLast%dDayNumtype_%d_userAllProductRatio'%(i,t)] = \
                   action['upLast%dDayNumtype_%d_userAllProductRatio'%(i,t)].apply(lambda x:0 if math.isinf(x) else x)

        # 删除不必要的特征/属性
        for t in range(1,7):
            # 该特征在之前的函数中已经添加过
            del action['upLast%dDayNumtype_%d'%(i,t)]
        for t in range(1,7):
            del action['upLast%dDayNumtype_%d_userAllProduct'%(i,t)]
        del action['upLast%dDayNumUserProductInteractive'%(i)]
        
        user_product_table = pd.merge(user_product_table, action, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
        user_product_table.to_csv(path, index=False, index_label=False)
    return user_product_table

# 比值类特征,操作次数的比值的排序, 输入经过提取操作次数比值特征的 user_product_table
def action_user_product_NumType_ratio_rank(user_product_table,last_time_list,end_date,flag):
    folder = ['train','test']
    path = "./tmp/%s/action_user_product_NumType_ratio_rank%s.csv" % (folder[flag],end_date)
    
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    
    print 'action_user_product_NumType_ratio_rank'
    for i in last_time_list:
        print "在购买的前 %d 天特征\n" % (i,)
######### 预测期前N天内对该商品的操作（购买、收藏、点击等行为）次数/访问次数的排序 #######
        # 6种行为比值的排序
        for t in range(1,7):
            record = user_product_table[['user_id','sku_id','upLast%dDayNumtype_%d_UserProductInteractiveRatio'%(i,t)]]
            record['upLast%dDayNumtype_%d_UserProductInteractiveRatio_rank'%(i,t)] =                     record.groupby('user_id')['upLast%dDayNumtype_%d_UserProductInteractiveRatio'%(i,t)].rank(
                        ascending=False, method='max')
            del record['upLast%dDayNumtype_%d_UserProductInteractiveRatio'%(i,t)]
            del user_product_table['upLast%dDayNumtype_%d_UserProductInteractiveRatio'%(i,t)]
            user_product_table = pd.merge(user_product_table, record, how='left', on=['user_id','sku_id'])
            
######### 预测期前N天内对该商品的操作（购买、收藏、点击等行为）次数/用户该操作总次数的排序，6种行为都包括 ############################       
        # 统计比值
        for t in range(1,7):
            record = user_product_table[['user_id','sku_id','upLast%dDayNumtype_%d_userAllProductRatio'%(i,t)]]
            record['upLast%dDayNumtype_%d_userAllProductRatio_rank'%(i,t)] =                     record.groupby('user_id')['upLast%dDayNumtype_%d_userAllProductRatio'%(i,t)].rank(
                        ascending=False, method='max')
            del record['upLast%dDayNumtype_%d_userAllProductRatio'%(i,t)]
            del user_product_table['upLast%dDayNumtype_%d_userAllProductRatio'%(i,t)]
            user_product_table = pd.merge(user_product_table, record, how='left', on=['user_id','sku_id'])
        user_product_table.to_csv(path, index=False, index_label=False)
        
    return user_product_table

# 比值类特征，时间天数比值
def action_user_product_TimeType_ratio(user_product_table,action_data,last_time_list,end_date,flag):
    folder = ['train','test']
    path = "./tmp/%s/action_user_product_TimeType_ratio%s.csv" % (folder[flag],end_date)
        
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    print 'action_user_product_TimeType_ratio'
    for i in last_time_list:
        print "在购买的前 %d 天特征\n" % (i,)
        start_date = (datetime.strptime(
            end_date,"%Y-%m-%d")-timedelta(days=i)).strftime("%Y-%m-%d")
        action_data = action_data[(action_data.time>=start_date)&(action_data.time<end_date)]
################ 预测期前N天 用户访问该商品的天数 / 用户总活跃天数 #######################################
        all_record = action_data[['user_id','sku_id','time']]
        all_record = all_record.drop_duplicates()
        # 计算用户访问该商品的天数
        time = all_record.groupby(['user_id','sku_id']).size().reset_index()
        # rename
        time.columns = ['user_id','sku_id','upLast%dDay_userInteractiveProductNumsDays'%i]
        # 计算用户的总活跃天数
        time_all = all_record.groupby(['user_id']).size().reset_index()
        # rename
        time_all.columns = ['user_id','upLast%dDay_userActiveNumsDays'%i]
        time = pd.merge(time,time_all,how='left',on='user_id')
        
        # 计算比值
        time['upLast%dDay_userInteractiveProductDaysRatio'%i] = \
            time['upLast%dDay_userInteractiveProductNumsDays'%i] / time['upLast%dDay_userActiveNumsDays'%i]
        
        del time['upLast%dDay_userInteractiveProductNumsDays'%i]
        del time['upLast%dDay_userActiveNumsDays'%i]
        
        user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
    
################ 预测期前N天 用户购买（点击/加购物车等）该商品的天数 / 用户总购买（点击/加购物车等）天数 #####################
        # 购买行为
        buy_record = action_data[action_data.type==4][['user_id','sku_id','time']]
        buy_record = buy_record.drop_duplicates()
        # 计算用户购买该商品的天数
        buy_time = buy_record.groupby(['user_id','sku_id']).size().reset_index()
        # rename
        buy_time.columns = ['user_id','sku_id','upLast%dDay_userBuyProductNumsDays'%i]
        # 计算用户的总购买天数
        buy_time_all = buy_record.groupby(['user_id']).size().reset_index()
        # rename
        buy_time_all.columns = ['user_id','upLast%dDay_userBuyNumsDays'%i]
        buy_time = pd.merge(buy_time,buy_time_all,how='left',on='user_id')
        # 计算比值
        buy_time['upLast%dDay_userBuyProductDaysRatio'%i] = \
            buy_time['upLast%dDay_userBuyProductNumsDays'%i] / buy_time['upLast%dDay_userBuyNumsDays'%i]
        del buy_time['upLast%dDay_userBuyProductNumsDays'%i]
        del buy_time['upLast%dDay_userBuyNumsDays'%i]
        
        user_product_table = pd.merge(user_product_table, buy_time, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
        
        # 浏览行为
        look_record = action_data[action_data.type==1][['user_id','sku_id','time']]
        look_record = look_record.drop_duplicates()
        # 计算用户浏览该商品的天数
        look_time = look_record.groupby(['user_id','sku_id']).size().reset_index()
        # rename
        look_time.columns = ['user_id','sku_id','upLast%dDay_userLookProductNumsDays'%i]
        # 计算用户的总浏览天数
        look_time_all = look_record.groupby(['user_id']).size().reset_index()
        # rename
        look_time_all.columns = ['user_id','upLast%dDay_userLookNumsDays'%i]
        look_time = pd.merge(look_time,look_time_all,how='left',on='user_id')
        # 计算比值
        look_time['upLast%dDay_userLookProductDaysRatio'%i] = \
            look_time['upLast%dDay_userLookProductNumsDays'%i] / look_time['upLast%dDay_userLookNumsDays'%i]
        del look_time['upLast%dDay_userLookProductNumsDays'%i]
        del look_time['upLast%dDay_userLookNumsDays'%i]
        
        user_product_table = pd.merge(user_product_table, look_time, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
       
        # 加入购物车行为
        add_record = action_data[action_data.type==2][['user_id','sku_id','time']]
        add_record = add_record.drop_duplicates()
        # 计算用户将该商品加入购物车的天数
        add_time = add_record.groupby(['user_id','sku_id']).size().reset_index()
        # rename
        add_time.columns = ['user_id','sku_id','upLast%dDay_userAddProductNumsDays'%i]
        # 计算用户的总加入购物车天数
        add_time_all = add_record.groupby(['user_id']).size().reset_index()
        # rename
        add_time_all.columns = ['user_id','upLast%dDay_userAddNumsDays'%i]
        add_time = pd.merge(add_time,add_time_all,how='left',on='user_id')
        # 计算比值
        add_time['upLast%dDay_userAddProductDaysRatio'%i] = \
            add_time['upLast%dDay_userAddProductNumsDays'%i] / add_time['upLast%dDay_userAddNumsDays'%i]
        del add_time['upLast%dDay_userAddProductNumsDays'%i]
        del add_time['upLast%dDay_userAddNumsDays'%i]
        
        user_product_table = pd.merge(user_product_table, add_time, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
        
        # 删除购物车行为
        del_record = action_data[action_data.type==3][['user_id','sku_id','time']]
        del_record = del_record.drop_duplicates()
        # 计算用户将该商品删除购物车的天数
        del_time = del_record.groupby(['user_id','sku_id']).size().reset_index()
        # rename
        del_time.columns = ['user_id','sku_id','upLast%dDay_userDelProductNumsDays'%i]
        # 计算用户的删除购物车天数
        del_time_all = del_record.groupby(['user_id']).size().reset_index()
        # rename
        del_time_all.columns = ['user_id','upLast%dDay_userDelNumsDays'%i]
        del_time = pd.merge(del_time,del_time_all,how='left',on='user_id')
        # 计算比值
        del_time['upLast%dDay_userDelProductDaysRatio'%i] = \
            del_time['upLast%dDay_userDelProductNumsDays'%i] / del_time['upLast%dDay_userDelNumsDays'%i]
        del del_time['upLast%dDay_userDelProductNumsDays'%i]
        del del_time['upLast%dDay_userDelNumsDays'%i]
        
        user_product_table = pd.merge(user_product_table, del_time, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
        
        # 关注行为
        favor_record = action_data[action_data.type==5][['user_id','sku_id','time']]
        favor_record = favor_record.drop_duplicates()
        # 计算用户关注该商品的天数
        favor_time = favor_record.groupby(['user_id','sku_id']).size().reset_index()
        # rename
        favor_time.columns = ['user_id','sku_id','upLast%dDay_userFavorProductNumsDays'%i]
        # 计算用户的总关注天数
        favor_time_all = favor_record.groupby(['user_id']).size().reset_index()
        # rename
        favor_time_all.columns = ['user_id','upLast%dDay_userFavorNumsDays'%i]
        favor_time = pd.merge(favor_time,favor_time_all,how='left',on='user_id')
        # 计算比值
        favor_time['upLast%dDay_userFavorProductDaysRatio'%i] = \
            favor_time['upLast%dDay_userFavorProductNumsDays'%i] / favor_time['upLast%dDay_userFavorNumsDays'%i]
        del favor_time['upLast%dDay_userFavorProductNumsDays'%i]
        del favor_time['upLast%dDay_userFavorNumsDays'%i]
        
        user_product_table = pd.merge(user_product_table, favor_time, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
        
        # 点击行为
        click_record = action_data[action_data.type==6][['user_id','sku_id','time']]
        click_record = click_record.drop_duplicates()
        # 计算用户点击该商品的天数
        click_time = click_record.groupby(['user_id','sku_id']).size().reset_index()
        # rename
        click_time.columns = ['user_id','sku_id','upLast%dDay_userClickProductNumsDays'%i]
        # 计算用户的总点击天数
        click_time_all = click_record.groupby(['user_id']).size().reset_index()
        # rename
        click_time_all.columns = ['user_id','upLast%dDay_userClickNumsDays'%i]
        click_time = pd.merge(click_time,click_time_all,how='left',on='user_id')
        # 计算比值
        click_time['upLast%dDay_userClickProductDaysRatio'%i] = \
            click_time['upLast%dDay_userClickProductNumsDays'%i] / click_time['upLast%dDay_userClickNumsDays'%i]
        del click_time['upLast%dDay_userClickProductNumsDays'%i]
        del click_time['upLast%dDay_userClickNumsDays'%i]
        
        user_product_table = pd.merge(user_product_table, click_time, how='left', on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
        
    user_product_table.to_csv(path, index=False, index_label=False)
    return user_product_table

# 比值类特征，时间天数比值的排序, 输入是提取了时间天数比值的 user_product_table
def action_user_product_TimeType_ratio_rank(user_product_table,last_time_list,end_date,flag):
    folder = ['train','test']
    path = "./tmp/%s/action_user_product_TimeType_ratio_rank%s.csv" % (folder[flag],end_date)
        
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    print 'action_user_product_TimeType_ratio_rank'
    for i in last_time_list:
        print "在购买的前 %d 天特征\n" % (i,)
################ 预测期前N天 用户访问该商品的天数 / 用户总活跃天数 的排序 #######################################
        time = user_product_table[['user_id','sku_id','upLast%dDay_userInteractiveProductDaysRatio'%i]]
        time['upLast%dDay_userInteractiveProductDaysRatio_rank'%i] = \
            time.groupby('user_id')['upLast%dDay_userInteractiveProductDaysRatio'%i].rank(
                ascending=False, method='max')
        del time['upLast%dDay_userInteractiveProductDaysRatio'%i]
        del user_product_table['upLast%dDay_userInteractiveProductDaysRatio'%i]
        user_product_table = pd.merge(user_product_table, time, how='left', on=['user_id','sku_id'])
    
################ 预测期前N天 用户购买（点击/加购物车等）该商品的天数 / 用户总购买（点击/加购物车等）天数 #####################
        # 购买行为
        buy_time=user_product_table[['user_id','sku_id','upLast%dDay_userBuyProductDaysRatio'%i]]
        buy_time['upLast%dDay_userBuyProductDaysRatio_rank'%i] = \
            buy_time.groupby('user_id')['upLast%dDay_userBuyProductDaysRatio'%i].rank(ascending=False,method='max')
        del buy_time['upLast%dDay_userBuyProductDaysRatio'%i]
        del user_product_table['upLast%dDay_userBuyProductDaysRatio'%i]
        user_product_table = pd.merge(user_product_table, buy_time, how='left', on=['user_id','sku_id'])
        
        # 浏览行为
        look_time = user_product_table[['user_id','sku_id','upLast%dDay_userLookProductDaysRatio'%i]]
        look_time['upLast%dDay_userLookProductDaysRatio_rank'%i] = \
            look_time.groupby('user_id')['upLast%dDay_userLookProductDaysRatio'%i].rank(ascending=False,method='max')
        del look_time['upLast%dDay_userLookProductDaysRatio'%i]
        del user_product_table['upLast%dDay_userLookProductDaysRatio'%i]
        user_product_table = pd.merge(user_product_table, look_time, how='left', on=['user_id','sku_id'])
       
        # 加入购物车行为
        add_time = user_product_table[['user_id','sku_id','upLast%dDay_userAddProductDaysRatio'%i]]
        add_time['upLast%dDay_userAddProductDaysRatio_rank'%i] = \
            add_time.groupby('user_id')['upLast%dDay_userAddProductDaysRatio'%i].rank(ascending=False,method='max')
        del add_time['upLast%dDay_userAddProductDaysRatio'%i]
        del user_product_table['upLast%dDay_userAddProductDaysRatio'%i]
        user_product_table = pd.merge(user_product_table, add_time, how='left', on=['user_id','sku_id'])
        
        # 删除购物车行为
        del_time = user_product_table[['user_id','sku_id','upLast%dDay_userDelProductDaysRatio'%i]]
        del_time['upLast%dDay_userDelProductDaysRatio_rank'%i] = \
            del_time.groupby('user_id')['upLast%dDay_userDelProductDaysRatio'%i].rank(ascending=False,method='max')
        del del_time['upLast%dDay_userDelProductDaysRatio'%i]
        del user_product_table['upLast%dDay_userDelProductDaysRatio'%i]
        user_product_table = pd.merge(user_product_table, del_time, how='left', on=['user_id','sku_id'])
        
        # 关注行为
        favor_time = user_product_table[['user_id','sku_id','upLast%dDay_userFavorProductDaysRatio'%i]]
        favor_time['upLast%dDay_userFavorProductDaysRatio_rank'%i] = \
            favor_time.groupby('user_id')['upLast%dDay_userFavorProductDaysRatio'%i].rank(ascending=False,method='max')
        del favor_time['upLast%dDay_userFavorProductDaysRatio'%i]
        del user_product_table['upLast%dDay_userFavorProductDaysRatio'%i]
        user_product_table = pd.merge(user_product_table, favor_time, how='left', on=['user_id','sku_id'])
        
        # 点击行为
        click_time = user_product_table[['user_id','sku_id', 'upLast%dDay_userClickProductDaysRatio'%i]]
        click_time['upLast%dDay_userClickProductDaysRatio_rank'%i] = \
            click_time.groupby('user_id')['upLast%dDay_userClickProductDaysRatio'%i].rank(ascending=False,method='max')
        del click_time['upLast%dDay_userClickProductDaysRatio'%i]
        del user_product_table['upLast%dDay_userClickProductDaysRatio'%i]
        user_product_table = pd.merge(user_product_table, click_time, how='left', on=['user_id','sku_id'])
    user_product_table.to_csv(path, index=False, index_label=False)
    return user_product_table

# 统计商品是否在前3天内有加购行为，设置 flag
def action_user_product_lastNday_add_flag(user_product_table,action_data,last_time_list,end_date,flag):
    folder = ['train','test']
    path = "./tmp/%s/action_user_product_lastNday_add_flag%s.csv" % (folder[flag],end_date)
        
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    print 'action_user_product_lastNday_add_flag'
    for i in last_time_list:
        print "在购买的前 %d 天特征\n" % (i,)
        start_date = (datetime.strptime(
            end_date,"%Y-%m-%d")-timedelta(days=i)).strftime("%Y-%m-%d")
        action_data = action_data[(action_data.time>=start_date)&(action_data.time<end_date)]
        record = action_data[action_data.userAdd==1][['sku_id', 'userAdd']]
        record['last%dDay_add_flag'%i] = 1
        user_product_table = pd.merge(user_product_table, record, how='left', on=['sku_id'])
        user_product_table = user_product_table.drop('userAdd',axis=1)
        user_product_table = user_product_table.fillna(0)
        user_product_table = user_product_table.drop_duplicates()
    user_product_table.to_csv(path, index=False, index_label=False)
    
    return user_product_table

# 统计用户对商品最后交互时间到预测期，增加小时和分钟，更加精确
def action_user_product_last_interact_record(user_product_table,action_data,end_date,flag,N=100):
    folder = ['train','test']
    path = "./tmp/%s/action_user_product_last_interact_record%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    print 'action_user_product_last_interact_record\n'

    action = action_data[action_data.time<end_date][['user_id','sku_id','time','timeHour','timeMinute']]
    #选出用户-商品对最后一条交互数据
    action = action.sort_values(by=['user_id','sku_id','time',],ascending=True).drop_duplicates(['user_id','sku_id'],'last')
    #【最后交互时间】
    action['userProductLastInteractTime'] = action['time'].apply(
        lambda x:(datetime.strptime(end_date,"%Y-%m-%d")-datetime.strptime(x,"%Y-%m-%d")).days-1)+(24-action['timeHour']-1)/24+(60-action['timeMinute'])/24/60
    #【最后交互时间是否是最后一天23点30后】
    action['userProductLastInteractIsLastDayAfter2330'] = action['userProductLastInteractTime'].apply(lambda x:1 if x<=30.0/60/24 else 0)
    #【最后交互时间是否是最后一天6点前】
    action['userProductLastInteractIsLastDayBefore0600']= action['userProductLastInteractTime'].apply(lambda x:1 if (x<=1.0)&(x>18.0/24) else 0)
    user_product_table = pd.merge(user_product_table,
            action[['user_id','sku_id','userProductLastInteractTime','userProductLastInteractIsLastDayAfter2330','userProductLastInteractIsLastDayBefore0600']],how='left',on=['user_id','sku_id'])
    #缺失值填充
    user_product_table['userProductLastInteractTime'] = user_product_table['userProductLastInteractTime'].fillna(100)
    user_product_table['userProductLastInteractIsLastDayAfter2330'] = user_product_table['userProductLastInteractIsLastDayBefore0600'].fillna(0)
    user_product_table.to_csv(path, index=False, index_label=False)
    return user_product_table

# 提取预测期前N天用户对cate=8商品的行为特征和交互天数
def up_love_cate8_brand(user_product_table,action_data,end_date,flag,list=[100]):
    folder = ['train','test']
    path = "./tmp/%s/up_love_cate8_brand%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    print 'up_love_cate8_brand'
    #时间窗特征：
    for i in list:
        start_date = datetime.strftime((datetime.strptime(end_date,"%Y-%m-%d")-timedelta(days=i)),"%Y-%m-%d")
        
        #【对cate8品牌操作次数】
        action=action_data[(action_data.time>=start_date)&(action_data.time<end_date)].groupby(
            ['user_id','brand']).sum().reset_index()
        action['upLast%dLookCate8BrandNum'%i]=action['userLook']
        action['upLast%dAddCate8BrandNum'%i]=action['userAdd']
        action['upLast%dDeleteCate8BrandNum'%i]=action['userDelete']
        action['upLast%dBuyCate8BrandNum'%i]=action['userBuy']
        action['upLast%dFavorCate8BrandNum'%i]=action['userFavor']
        action['upLast%dClickCate8BrandNum'%i]=action['userClick']
        #【对cate8品牌操作数占用户对cate8总数比例】
        #用户总操作次数
        user_action=action_data[(action_data.time>=start_date)&(action_data.time<end_date)].groupby(
            ['user_id']).sum().reset_index()
        user_action = user_action.drop(['brand'],axis=1)
        action = pd.merge(action[['user_id','brand','upLast%dLookCate8BrandNum'%i,'upLast%dAddCate8BrandNum'%i,
                    'upLast%dDeleteCate8BrandNum'%i,'upLast%dBuyCate8BrandNum'%i,'upLast%dFavorCate8BrandNum'%i,
                    'upLast%dClickCate8BrandNum'%i]],user_action,how='left',on=['user_id'])
        action['upLast%dLookCate8BrandNumRatio'%i]=action['upLast%dLookCate8BrandNum'%i]/action['userLook']
        action['upLast%dAddCate8BrandNumRatio'%i]=action['upLast%dAddCate8BrandNum'%i]/action['userAdd']
        action['upLast%dDeleteCate8BrandNumRatio'%i]= action['upLast%dDeleteCate8BrandNum'%i]/action['userDelete']
        action['upLast%dBuyCate8BrandNumRatio'%i]=action['upLast%dBuyCate8BrandNum'%i]/action['userBuy']
        action['upLast%dFavorCate8BrandNumRatio'%i]=action['upLast%dFavorCate8BrandNum'%i]/action['userFavor']
        action['upLast%dClickCate8BrandNumRatio'%i]= action['upLast%dClickCate8BrandNum'%i]/action['userClick']
        user_product_table = pd.merge(user_product_table,action[['user_id','brand','upLast%dLookCate8BrandNumRatio'%i,'upLast%dAddCate8BrandNumRatio'%i,
                    'upLast%dDeleteCate8BrandNumRatio'%i,'upLast%dBuyCate8BrandNumRatio'%i,'upLast%dFavorCate8BrandNumRatio'%i,
                    'upLast%dClickCate8BrandNumRatio'%i]],how='left',on=['user_id','brand'])
        user_product_table = user_product_table.fillna(0)
        #【对cate8品牌的关注天数】
        action = action_data[(action_data.time>=start_date)&(action_data.time<end_date)][
            ['user_id','brand','time']].drop_duplicates()
        action['upLast%dCate8BrandFocusDays'%i] = 1
        action = action.groupby(['user_id','brand']).sum().reset_index()
        #【对cate8品牌关注天数占用户对cate8总数比例】
        #用户关注cate8天数
        user_action=action_data[(action_data.time>=start_date)&(action_data.time<end_date)][
            ['user_id','brand','time']].drop_duplicates()
        user_action['activeDays'] = 1
        user_action = user_action.groupby(['user_id']).sum().reset_index().drop(['brand'],axis=1)
        action = pd.merge(action[['user_id','brand','upLast%dCate8BrandFocusDays'%i]],user_action,how='left',on=['user_id'])
        action['upLast%dCate8BrandFocusDaysRatio'%i]=action['upLast%dCate8BrandFocusDays'%i]/action['activeDays']
        user_product_table = pd.merge(user_product_table,action[['user_id','brand','upLast%dCate8BrandFocusDaysRatio'%i]],
                                      how='left',on=['user_id','brand'])
        user_product_table = user_product_table.fillna(0)
    del user_product_table['brand']
    user_product_table.to_csv(path, index=False, index_label=False)
       
    return user_product_table
    
# 提取用户对商品的6种操作行为的次数和关注天数
def up_love_cate8_product(user_product_table,action_data,end_date,flag,list=[100]):
    folder = ['train','test']
    path = "./tmp/%s/up_love_cate8_product%s.csv" % (folder[flag],end_date)
    if os.path.exists(path):
        print "There is csv!!!",path
        user_product_table  = pd.read_csv(path)
        return user_product_table
    print 'up_love_cate8_product'
    #时间窗特征：
    for i in list:
        start_date = datetime.strftime((datetime.strptime(end_date,"%Y-%m-%d")-timedelta(days=i)),"%Y-%m-%d")
        
        #【对cate8品牌商品次数】
        action=action_data[(action_data.time>=start_date)&(action_data.time<end_date)].groupby(
            ['user_id','sku_id']).sum().reset_index()
        action['upLast%dLookCate8ProductNum'%i]=action['userLook']
        action['upLast%dAddCate8ProductNum'%i]=action['userAdd']
        action['upLast%dDeleteCate8ProductNum'%i]=action['userDelete']
        action['upLast%dBuyCate8ProductNum'%i]=action['userBuy']
        action['upLast%dFavorCate8ProductNum'%i]=action['userFavor']
        action['upLast%dClickCate8ProductNum'%i]=action['userClick']
        #【对cate8品牌操作数占用户对cate8总数比例】
        #用户总操作次数
        user_action=action_data[(action_data.time>=start_date)&(action_data.time<end_date)].groupby(
            ['user_id']).sum().reset_index()
        user_action = user_action.drop(['sku_id'],axis=1)
        action = pd.merge(action[['user_id','sku_id','upLast%dLookCate8ProductNum'%i,'upLast%dAddCate8ProductNum'%i,
                    'upLast%dDeleteCate8ProductNum'%i,'upLast%dBuyCate8ProductNum'%i,'upLast%dFavorCate8ProductNum'%i,
                    'upLast%dClickCate8ProductNum'%i]],user_action,how='left',on=['user_id'])
        action['upLast%dLookCate8ProductNumRatio'%i]=action['upLast%dLookCate8ProductNum'%i]/action['userLook']
        action['upLast%dAddCate8ProductNumRatio'%i]=action['upLast%dAddCate8ProductNum'%i]/action['userAdd']
        action['upLast%dDeleteCate8ProductNumRatio'%i]= action['upLast%dDeleteCate8ProductNum'%i]/action['userDelete']
        action['upLast%dBuyCate8ProductNumRatio'%i]=action['upLast%dBuyCate8ProductNum'%i]/action['userBuy']
        action['upLast%dFavorCate8ProductNumRatio'%i]=action['upLast%dFavorCate8ProductNum'%i]/action['userFavor']
        action['upLast%dClickCate8ProductNumRatio'%i]= action['upLast%dClickCate8ProductNum'%i]/action['userClick']
        user_product_table = pd.merge(user_product_table,action[['user_id','sku_id','upLast%dLookCate8ProductNumRatio'%i,'upLast%dAddCate8ProductNumRatio'%i,
                    'upLast%dDeleteCate8ProductNumRatio'%i,'upLast%dBuyCate8ProductNumRatio'%i,'upLast%dFavorCate8ProductNumRatio'%i,
                    'upLast%dClickCate8ProductNumRatio'%i]],how='left',on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
        #【对cate8品牌的关注天数】
        action = action_data[(action_data.time>=start_date)&(action_data.time<end_date)][
            ['user_id','sku_id','time']].drop_duplicates()
        action['upLast%dCate8ProductFocusDays'%i] = 1
        action = action.groupby(['user_id','sku_id']).sum().reset_index()
        #【对cate8品牌关注天数占用户对cate8总数比例】
        #用户关注cate8天数
        user_action=action_data[(action_data.time>=start_date)&(action_data.time<end_date)][
            ['user_id','sku_id','time']].drop_duplicates()
        user_action['activeDays'] = 1
        user_action = user_action.groupby(['user_id']).sum().reset_index().drop(['sku_id'],axis=1)
        action = pd.merge(action[['user_id','sku_id','upLast%dCate8ProductFocusDays'%i]],user_action,how='left',on=['user_id'])
        action['upLast%dCate8ProductFocusDaysRatio'%i]=action['upLast%dCate8ProductFocusDays'%i]/action['activeDays']
        user_product_table = pd.merge(user_product_table,action[['user_id','sku_id','upLast%dCate8ProductFocusDaysRatio'%i]],
                                      how='left',on=['user_id','sku_id'])
        user_product_table = user_product_table.fillna(0)
    del user_product_table['brand']
    user_product_table.to_csv(path, index=False, index_label=False)
       
    return user_product_table


# 生成标签
def labels(all_action,start_date, end_date):
    actions = all_action[(all_action.time >= start_date) & (all_action.time < end_date)]
    actions = actions[actions['type'] == 4]
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    actions['label'] = 1
    actions = actions[['user_id', 'sku_id', 'label']]
    return actions

# 随即选取一定比例 正负样本
def random_select(action, ratio):
    pos_sample = action[action['label'] ==1] 
    neg_sample = action[action['label'] ==0] 
    print " 生成特征 负样本数目:\n" ,neg_sample.shape
    print " 生成特征 正样本数目:\n" ,pos_sample.shape
    pos_num = float(pos_sample.shape[0])
    neg_num = float(neg_sample.shape[0])
    print " 生成特征 正/负样本比例:" ,pos_num/neg_num
    # random select
    sample_num = pos_num/ratio
    set_ratio = sample_num/neg_num
    print " 采样比例set_ratio:",set_ratio
    df_random_res = neg_sample.sample(frac = set_ratio)    
    print " 采样之后的 正/负样本比例:" ,pos_num/df_random_res.shape[0]
    # concat    
    res = pd.concat([pos_sample,df_random_res],axis=0)    
    return res


#  数据集生成 feat_set Train set && test set
# 输入 all_action
# train_start_date, train_end_date 时间窗口
# label_start_date, label_end_date 标签时间窗口
# isTrain=True 默认生成 训练集 False 表示生成线上测试集合 
# 生成线下 train set 的设置: 默认
# 生成 test 集 的设置: isTrain=False 
# ratio=0.1 正负样本比例
def feat_set(all_action,train_start_date, train_end_date, label_start_date= '2016-02-28', 
             label_end_date= '2016-03-28',ratio=0.1,isTrain=True,flag=0):
    feat_list=[]
    ############################ 构建预测 用户-商品 pair ######################
    # 第一种方法是 使用滑动时间窗 出现的 用户-商品 pair
    print "根据时间窗口(%s - %s) 包含的 用户-商品项" % (train_start_date,train_end_date)
    if isTrain:
        print "随机选取得到的 正/负样本比例是:", ratio
    user_product= gen_user_product_table(all_action,train_start_date, train_end_date)
    
    ############################ 用户特征 ######################
    print "***** step1 用户表生成 ******\n"
    user_table = user_product.drop('sku_id',axis=1)
    user_table = user_table.drop_duplicates().reset_index()
    user_table.drop('index',axis=1,inplace=True)
    print "1.1-用户表生成数目:\n",user_table.shape
    
    print "---1.2 创建当前用户总体特征 user_feat1---\n"
    user_feat1 = user_active_all_statistics(user_table, all_action, train_end_date,flag)
    user_feat1.head()
    
    print "---1.3 当前用户在 end_date 前user_active_statistics_product特征---\n"
    user_feat2 = user_active_statistics_product(user_table,all_action,train_end_date,flag)

    user_feat3 = user_buy_and_repeat_buy_flag(user_table,all_action,train_end_date,flag)
    user_feat4 = user_last1_activeDay_interact(user_table,all_action,train_end_date,flag)
    user_feat5 = user_last_interact_record(user_table,all_action,train_end_date,flag)
    user_feat6 = user_ActiveDay_daily_time_num(user_table,all_action,train_end_date,flag)
    
    user_feat7 = user_lastNDay_InteractTime(user_table,all_action,train_end_date,user_feat6,flag,7)
    user_feat8 = user_Cate8_feature(user_table,all_action,train_end_date,flag,7)
    user_feat9 = user_last1_activeDay_interact_times(user_table, all_action, train_end_date, flag)
    
    print "---1.4 用户特征组合---\n"
    user_all_feat = pd.merge(user_feat1, user_feat2, how='left', on='user_id')
    user_all_feat = pd.merge(user_all_feat, user_feat3, how='left', on='user_id')
    user_all_feat = pd.merge(user_all_feat, user_feat4, how='left', on='user_id')
    user_all_feat = pd.merge(user_all_feat, user_feat5, how='left', on='user_id')
    user_all_feat = pd.merge(user_all_feat, user_feat6, how='left', on='user_id')
    user_all_feat = pd.merge(user_all_feat, user_feat7, how='left', on='user_id')
    user_all_feat = pd.merge(user_all_feat, user_feat8, how='left', on='user_id')
    user_all_feat = pd.merge(user_all_feat, user_feat9, how='left', on='user_id')
    
    print " ---1.5 user_feat Merge 基础用户特征 (age sex level register)---\n"
    users=clean_user_data(JData_User_Path)
    basic_user_feat = ori_user_feat(users)
    user_all_feat = pd.merge(user_all_feat, basic_user_feat, how='left', on='user_id')
    
    ############################ 商品特征 ######################
    print "***** step2 商品表生成 *****\n" 
    product_table = user_product.drop('user_id',axis=1)
    product_table = product_table.drop_duplicates().reset_index()
    product_table.drop('index',axis=1,inplace=True)
    print "2.1 商品表生成数目:\n",product_table.shape
    
    print "---2.2 创建当前商品 product_feat1---\n"
    product_feat1 = product_active_all_statistics(product_table,all_action,train_end_date,flag)
    
    print "---2.3 当前用户在product的 end_date 前的行为特征 product_feat2---\n"
    last_time_list2 = [1,2,3,5,7]
    product_feat2 = product_last_days_statistics(product_table,all_action,last_time_list2,train_end_date,flag)
    
    print "---2.4 创建product对单个用户的统计特征 product_feat3---\n"
    product_feat3 = product_active_statistic_user(product_table,all_action,train_end_date,flag)
    
    print "---2.5 商品特征gen_all_product_feat---\n"
    action_product_feat = gen_all_product_feat()
    product_all_feat = pd.merge(product_feat1, action_product_feat, how='left', on='sku_id')
    product_all_feat = pd.merge(product_all_feat, product_feat2, how='left', on='sku_id')
    product_all_feat = pd.merge(product_all_feat, product_feat3, how='left', on='sku_id')

    product_all_feat=product_all_feat.fillna(0)
    
    print "---2.6 product_feat Merge 基础商品特征 (商品评论特征gen_comment_feat)---\n"
    comment_feat = gen_comment_feat(train_end_date)
    product_all_feat = pd.merge(product_all_feat, comment_feat, how='left', on='sku_id')    
            
    ############################ 用户-商品 特征 ######################
    
    print "***** step3 用户-商品表生成 *****\n" 
    user_product_table = user_product

    print "---3.1 用户-商品 特征 ---\n"
    last_day_list3 = [1,2,3,5,7]
    user_product_feat1 = get_last_days_action(
        user_product_table,all_action,last_day_list3,train_end_date,flag)
    user_product_feat2 = get_last_days_action_rank(
        user_product_table, all_action, last_day_list3,train_end_date,flag)
    
    user_product_feat3 = action_get_interactive_days(
        user_product_table, all_action,train_end_date,flag)
    user_product_feat4 = action_get_interactive_days_2(
        user_product_table, all_action, train_end_date, flag)
    user_product_feat_tmp1 = pd.merge(user_product_feat3, user_product_feat4, how='left', on=['user_id', 'sku_id'])
    user_product_feat5 = action_get_interactive_days_rank(user_product_feat_tmp1, train_end_date, flag)
    
    user_product_feat6 = action_user_product_NumType_ratio(
        user_product_table,all_action,last_day_list3,train_end_date,flag)
    user_product_feat7 = action_user_product_NumType_ratio_rank(
        user_product_feat6, last_day_list3, train_end_date, flag)
    
    user_product_feat8 = action_user_product_TimeType_ratio(
        user_product_table,all_action, last_day_list3,train_end_date,flag)
    user_product_feat9 = action_user_product_TimeType_ratio_rank(
        user_product_feat8, last_day_list3, train_end_date, flag)
    
    last_day_list4 = [1,2,3]
    user_product_feat10 = action_user_product_lastNday_add_flag(
        user_product_table,all_action, last_day_list4,train_end_date,flag)
    user_product_feat11 = action_user_product_last_interact_record(
        user_product_table, all_action, train_end_date, flag)
    
    action = all_action.drop_duplicates()
    user_product_table2 = action[(action.time >= train_start_date) & (action.time < train_end_date)]
    user_product_table2 = user_product_table2[['user_id', 'sku_id','type','brand']]
    user_product_table2 = user_product_table2.groupby(['user_id', 'sku_id','brand'], as_index=False).sum()
    del user_product_table2['type']
    
    user_product_feat12 = up_love_cate8_brand(user_product_table2, all_action, train_end_date, flag)
    user_product_feat13 = up_love_cate8_product(user_product_table2, all_action, train_end_date, flag)
    
    user_product_feat = pd.merge(user_product_feat1, user_product_feat2, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat3, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat4, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat5, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat6, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat7, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat8, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat9, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat10, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat11, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat12, how='left', on=['user_id', 'sku_id'])
    user_product_feat = pd.merge(user_product_feat, user_product_feat13, how='left', on=['user_id', 'sku_id'])
    
    # 保存特征名字，后续可以根据特征在 feat_list中的编号来筛选重要的特征
    feat_list.append(list(user_feat1.columns))
    feat_list.append(list(user_feat2.columns))
    feat_list.append(list(user_feat3.columns))
    feat_list.append(list(user_feat4.columns))
    feat_list.append(list(user_feat5.columns))
    feat_list.append(list(user_feat6.columns))
    feat_list.append(list(user_feat7.columns))
    feat_list.append(list(user_feat8.columns))
    feat_list.append(list(user_feat9.columns))
    feat_list.append(list(product_feat1.columns))
    feat_list.append(list(product_feat2.columns))
    feat_list.append(list(product_feat3.columns))
    feat_list.append(list(comment_feat.columns))
    feat_list.append(list(user_product_feat1.columns))
    feat_list.append(list(user_product_feat2.columns))
    feat_list.append(list(user_product_feat3.columns))
    feat_list.append(list(user_product_feat4.columns))
    feat_list.append(list(user_product_feat5.columns))
    feat_list.append(list(user_product_feat6.columns))
    feat_list.append(list(user_product_feat7.columns))
    feat_list.append(list(user_product_feat8.columns))
    feat_list.append(list(user_product_feat9.columns))
    feat_list.append(list(user_product_feat10.columns))
    feat_list.append(list(user_product_feat11.columns))
    feat_list.append(list(user_product_feat12.columns))
    feat_list.append(list(user_product_feat13.columns))
    
    ############################ 特征组合 ######################
    print "***** step4 特征组合 *****\n" 
    print "4.1 all_feat 组合 用户特征  user_all_feat\n" 
    all_feat = pd.merge(user_product, user_all_feat, how='left', on='user_id')
    print "4.2 all_feat 组合 商品特征  product_all_feat\n" 
    all_feat = pd.merge(all_feat, product_all_feat, how='left', on='sku_id')
    print "4.3 all_feat 组合特征  user_product_feat\n" 
    all_feat = pd.merge(all_feat, user_product_feat, how='left', on=['user_id', 'sku_id'])
    
    ############################ 生成标签 ######################
    
    if isTrain:
        print "***** 生成 训练集 标签(含有全部 cate) *****\n" 
        label = labels(all_action,label_start_date, label_end_date)
            
        all_feat = pd.merge(all_feat, label, how='left',  on=['user_id', 'sku_id'])
        all_feat = all_feat.fillna(0)
        ########按比例随机选取 训练数据
        all_feat = random_select(all_feat, ratio)
        
        train_label = all_feat['label'].copy()
        del all_feat['label']
        user_sku = all_feat[['user_id', 'sku_id']].copy()
        del all_feat['user_id']
        del all_feat['sku_id']
        print "***** 生成训练特征 *****\n" 
        return user_sku,all_feat,train_label,label,feat_list
    else:
        print "***** 生成线上提交特征 *****\n" 
        user_sku = all_feat[['user_id', 'sku_id']].copy()
        del all_feat['user_id']
        del all_feat['sku_id']
        return user_sku,all_feat
    
def make_train_set(all_action,train_start_date, train_end_date, label_start_date, label_end_date):
    return feat_set(all_action,train_start_date, train_end_date, label_start_date, label_end_date,isTrain=True,flag=0)

def make_test_set(all_action,train_start_date, train_end_date):
    return feat_set(all_action,train_start_date, train_end_date,isTrain=False,flag=1)

# saveFeat=False 默认不保存特征 以节省时间
def create_train_set(all_action,train_zip):
    user_sku_chunks = []
    train_data_chunks = []
    train_label_chunks = []
    train_origin_label_chunks = []

    for i,(train_start_date,train_end_date,label_start_date,label_end_date) in enumerate(train_zip):
        print ('*******第%d个考察期:%s --->  %s*******' % (i+1,train_start_date, train_end_date))
        try:
            user_sku_tmp, train_data_tmp, train_label_tmp,train_origin_label_tmp, feat_list= make_train_set(
                all_action,train_start_date, train_end_date, label_start_date, label_end_date)
            user_sku_chunks.append(user_sku_tmp)
            train_data_chunks.append(train_data_tmp)
            train_label_chunks.append(train_label_tmp)
            train_origin_label_chunks.append(train_origin_label_tmp)
        except:
            print ('%s-%s-stop' % (train_start_date, train_end_date))
            return

    train_user_sku = pd.concat(user_sku_chunks, ignore_index=True)
    train_data = pd.concat(train_data_chunks, ignore_index=True)
    train_label = pd.concat(train_label_chunks, ignore_index=True)
    train_origin_label = pd.concat(train_origin_label_chunks, ignore_index=True)    
    
    return train_user_sku,train_data,train_label,train_origin_label, feat_list

# 特征筛选
#　delfeatlist 存储需要删除的特征名
def delfeat(data,delfeatlist):
    tmp=data.drop(delfeatlist,axis=1)
    print tmp.shape
    return tmp

if __name__ == '__main__':
    # 载入数据 
    # 用户数据
    JData_User_Path = "JData/JData_User.csv" 
    # 行为数据
    print '{} loading all action data...'.format(datetime.now())
    all_action = pd.read_csv('JData/all_action_with_commit_user_clean_all_time.csv')
    # 生成行为数据涉及的全部商品数目 ./tmp/action_product.csv
    gen_all_product_from_action(all_action)
    # 生成训练集
    # 取前十二天
    train_start_date_list = ['2016-04-01', '2016-03-18', '2016-03-30', '2016-03-28', '2016-03-26', '2016-03-24', '2016-03-22', '2016-03-20']
    train_end_date_list = ['2016-04-09','2016-03-26','2016-04-07','2016-04-05','2016-04-03','2016-04-01','2016-03-30','2016-03-28']
    label_start_date_list = ['2016-04-09','2016-03-26','2016-04-07','2016-04-05','2016-04-03','2016-04-01','2016-03-30','2016-03-28']
    label_end_date_list = ['2016-04-14','2016-03-31','2016-04-12','2016-04-10','2016-04-08','2016-04-06','2016-04-04','2016-04-02']
    train_zip = zip(train_start_date_list,train_end_date_list,label_start_date_list,label_end_date_list)
    
    print '{} generating train dataset'.format(datetime.now())
    train_user_sku,train_data,train_label,train_origin_label, feat_list=create_train_set(
        all_action,train_zip)
    print '{} finish'.format(datetime.now())
    
    delfeatlist = []
    # 选择需要删除的特征编号
    idx = [14, 20, 21,22,24]
    for i in idx:
        delfeatlist += feat_list[i][2:]
    # 执行特征筛选
    train_data_select=delfeat(train_data,delfeatlist)
    # 保存训练集
    print "保存 train_user_sku_gen.csv"
    train_user_sku.to_csv('tmp/train_user_sku_gen.csv', index=False, index_label=False)
    print "保存 train_data_select.csv"
    train_data_select.to_csv('tmp/train_data_select.csv', index=False, index_label=False)
    print "保存 train_label_gen.csv"
    train_label.to_csv('tmp/train_label_gen.csv', index=False, index_label=False)
    print "保存 train_origin_label_gen.csv"
    train_origin_label.to_csv('tmp/train_origin_label_gen.csv', index=False, index_label=False)
    
    # 模型训练
    # xgboost模型训练   
    xgb_model = XGBClassifier(
            learning_rate =0.1,
            n_estimators=420,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
    print '{} start to train xgboost model'.format(datetime.now())
    # 使用所有训练集训练模型，得到最终的模型
    xgb_model = xgb_model.fit(train_data_select.values, train_label.values,eval_metric='auc')    
    print '{} finish'.format(datetime.now())
    
    # 保存xgboost模型
   
    modelpath ='model/user_sku_xgb_model.pkl'
    print "保存模型:",modelpath
    joblib.dump(xgb_model,modelpath)
    # lightGBM模型训练
    gbm = lgb.LGBMClassifier(learning_rate=0.1,num_leaves=6,max_depth=5,colsample_bytree=0.9,subsample=1,min_child_weight=1,
                             n_estimators=1615)
    print '{} start to train lightGBM model'.format(datetime.now())
    gbm.fit(train_data_select,train_label,eval_metric='auc')
    print '{} finish'.format(datetime.now())
    
    # 保存lightGBM模型
    modelpath ='model/user_sku_gbm_model.pkl'
    print "保存模型:",modelpath
    joblib.dump(gbm,modelpath)





