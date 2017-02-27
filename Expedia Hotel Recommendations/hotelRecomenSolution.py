# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:23:53 2017

@author: cai

Expedia Hotel Recommendation比赛数据处理
"""

import numpy as np
import pandas as pd

# 记录程序运行时间
import time
start_time = time.time()

destinations = pd.read_csv("destinations.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

#print(train.shape)
#print(test.shape)
# 查看训练数据中 hotel_cluster，即每个集群中酒店的个数，hotel_cluster就是待预测值
#print(train['hotel_cluster'].value_counts())

# 创建包含所有测试数据集用户id唯一值的集合
#==============================================================================
# test_ids = set(test.user_id.unique())
# # 创建包含所有训练数据集用户id唯一值的集合
# train_ids = set(train.user_id.unique())
# # 计算出有多少测试数据中的用户id在训练集中的用户id中
# intersection_count = len(test_ids & train_ids)
# # 看看匹配的数目和测试数据集的用户id总数是否一样
# print(intersection_count == len(test_ids))
#==============================================================================

# 将data_time列从object转换成datatime值，这能使它比日期工作起来简单许多。
# 将year和month从data_time提取出来并赋值到它们的列。
train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month

#import random

#unique_users = train.user_id.unique()
# 随机抽取10000个用户的数据
#sel_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), 10000)) ]
#sel_train = train[train.user_id.isin(sel_user_ids)]
# 创建新的训练集t1和交叉验证集t2,分别包含2014年8月前的数据和8月后的数据
#t1 = sel_train[((sel_train.year == 2013) | ((sel_train.year == 2014) & (sel_train.month < 8)))]
#t2 = sel_train[((sel_train.year == 2014) & (sel_train.month >= 8))]
# t2 shape = 114522 * 26
#print(t2.shape)
# 如果is_booking的值是0，表明是一次点击，值是1表明是一个预定。
#测试数据集值包含预定事件，所以我们也需要将t2简化成只包含预定
#t2 = t2[t2.is_booking == True]
# t2 shape = 8185 * 26
#print(t2.shape)
t1 = train
t2 = test
print(t2.shape)

# 展示训练数据集中一个包含五个最常见的集群的列表
most_common_clusters = list(train.hotel_cluster.value_counts().head().index)

# 创建一个元素和t2的行数一样多的列表，每个元素等同于most_common_clusters
#predictions = [most_common_clusters for i in range(t2.shape[0])]

# 计算Mean Average Precision
import average_precision as ap

#target = [[l] for l in t2["hotel_cluster"]]
#print(ap.mapk(target, predictions, k=5))

# 找是否有和hotel_cluster非常相关的元素
#print(train.corr()["hotel_cluster"])

from sklearn.decomposition import PCA
# 保留数据集中的3列数据
pca = PCA(n_components=3)
# 将d1-d149列转换成3列
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small["srch_destination_id"] = destinations["srch_destination_id"]

def calc_fast_features(df):
    """
    生成新的像data_time的数据特征列。
    删除像data_time的非数据列。
    从dest_small添加特征。
    """
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")

    props = {}
    # 生成新的像data_time的数据特征列, 如"month", "day", "hour", "minute", "dayofweek", "quarter"
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)

    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    #print(carryover)
    # 结合其他非时间特征
    for prop in carryover:
        props[prop] = df[prop]
        
    date_props = ["month", "day", "dayofweek", "quarter"]
    # 生成跟住入和离开时间的时间特征生成跟住入和离开时间的时间特征
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')

    ret = pd.DataFrame(props)
    # 从dest_small添加特征
    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret

df = calc_fast_features(t1)
# 用-1替换所有的缺失值
df.fillna(-1, inplace=True)

# 用随机森林算法来生成预测, 使用3折交叉验证使用训练数据集来生成一个可靠的误差估计
#predictors = [c for c in df.columns if c not in ["hotel_cluster"]]
#==============================================================================
# from sklearn import cross_validation
# from sklearn.ensemble import RandomForestClassifier
# 
# clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
# scores = cross_validation.cross_val_score(clf, df[predictors], df['hotel_cluster'], cv=3)
# print(scores)
#==============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from itertools import chain
# 再次训练随机森林，但是每一个森林都将只预测一个单独的酒店集群，
#为了速度我们将使用2折交叉验证，而且每个标签只训练10棵树
#==============================================================================
# all_probs = []
# unique_clusters = df["hotel_cluster"].unique ()
# for cluster in unique_clusters:
#     df["target"] = 1
#     df["target"][df["hotel_cluster"] != cluster] = 0
#     predictors = [col for col in df if col not in ['hotel_cluster' , "target"]]
#     probs = []
#     # 使用2折交叉验证训练一个随机森林分类器,cv包括两个列表，分别是将训练集分成两部分，包含的是行号
#     cv = KFold(len(df["target"]), n_folds = 2)
#     clf = RandomForestClassifier(n_estimators = 10 , min_weight_fraction_leaf = 0.1)
#     for i, (tr,te) in enumerate(cv):
#         # tr for trainDataset, te for testDataset
#         #print(i, tr, te)
#         clf.fit(df[predictors].iloc[tr], df["target"].iloc[tr])
#         preds = clf.predict_proba(df[predictors].iloc[te])
#         #print(preds)
#         probs.append([p[1] for p in preds])
#         #print(probs)
#     full_probs = chain.from_iterable(probs)
#     #print(full_probs)
#     all_probs.append(list(full_probs))
# 
# prediction_frame = pd.DataFrame(all_probs).T
# prediction_frame.columns = unique_clusters
# def find_top_5(row):
#     return list(row.nlargest(5).index)
# 
#==============================================================================
#==============================================================================
# preds  =  []
# # 对于每一行，找出5个最大的概率并且将hotel_cluster的值赋值给预测
# for index, row in prediction_frame.iterrows():
#     preds.append(find_top_5(row))
# 
# print(ap.mapk([[l] for l in t2.iloc["hotel_cluster"]], preds, k = 5))
# 
#==============================================================================
# 聚合'orig_destination_distance'会为每一个目标找到最受欢迎的酒店集群，
#然后我们就能够预测某个用户搜索的目标是最受欢迎的酒店集群中的一个
def make_key(items):
    return "_".join([str(i) for i in items])

match_cols = ["srch_destination_id"]
cluster_cols = match_cols + ['hotel_cluster']

# 通过srch_destination_id，hotel_cluster将t1分组
groups = t1.groupby(cluster_cols)

top_clusters = {}
for name, group in groups:
    clicks = len(group.is_booking[group.is_booking == False])
    bookings = len(group.is_booking[group.is_booking == True])

    score = bookings + .15 * clicks

    clus_name = make_key(name[:len(match_cols)])
    if clus_name not in top_clusters:
        top_clusters[clus_name] = {}
    top_clusters[clus_name][name[-1]] = score
#print('top_clusters',top_clusters)

import operator
# 变换 top_clusters 这个字典来找到每个'srch_destination_id'的前五个酒店集群
cluster_dict = {}
for n in top_clusters:
    tc = top_clusters[n]
    # 找到键中的最高的5个集群
    top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
    cluster_dict[n] = top
# 开始做预测 
preds = []
for index, row in t2.iterrows():
    key = make_key([row[m] for m in match_cols])
    if key in cluster_dict:
        preds.append(cluster_dict[key])
    else:
        preds.append([])
#print('preds=', preds)    
# 使用mapk函数来计算准确率          
#print(ap.mapk([[l] for l in t2.iloc["hotel_cluster"]], preds, k = 5))
               

match_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']
# 将训练数据集以匹配的列切分分组
groups = t1.groupby(match_cols)
#print(group)
def generate_exact_matches(row, match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    clus = list(set(group.hotel_cluster))
    return clus

exact_matches = []
for i in range(t2.shape[0]):
    exact_matches.append(generate_exact_matches(t2.iloc[i], match_cols))
# 使用f5函数只选取唯一的预测，按顺序排列
def f5(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

full_preds = [f5(exact_matches[p] + preds[p] + most_common_clusters)[:5] for p in range(len(preds))]
#print(ap.mapk([[l] for l in t2.iloc['hotel_cluster']], preds, k = 5))
# save to csv file
write_p = [" ".join([str(l) for l in p]) for p in full_preds]
write_frame = ["{0},{1}".format(t2["id"][i], write_p[i]) for i in range(len(full_preds))]
write_frame = ["id,hotel_cluster"] + write_frame
with open("predictions.csv", "w+") as f:
    f.write("\n".join(write_frame))

end = time.time()
print('totally time is ', end-start_time)
  
