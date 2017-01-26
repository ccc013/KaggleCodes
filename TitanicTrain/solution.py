#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2017/1/26 20:48
@Author  : cai

"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
sns.set_style('whitegrid')
# 读入csv数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# print(train_df)

# 数据预处理
# 去除不必要的字段
train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
# print(train_df)

# 对属性Embarked缺失数据使用最多的属性值'S'填充
train_df['Embarked'] = train_df['Embarked'].fillna('S')
# print(train_df['Embarked'])
# 绘图
sns.factorplot('Embarked', 'Survived', data=train_df, size=4, aspect=3)
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 5))
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue='Embarked', data=train_df, ax=axis2)
embark_perc = train_df[['Embarked', 'Survived']].groupby(['Embarked'],
            as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)

# 将Embark属性值转换为数字值特征（one-hot encoding)
embark_dummies_train = pd.get_dummies(train_df['Embarked'])
# print(embark_dummies_train)
# 去掉属性值'S'
embark_dummies_train.drop(['S'], axis=1, inplace=True)
embark_dummies_test = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)
train_df = train_df.join(embark_dummies_train)
test_df = test_df.join(embark_dummies_test)
train_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)
# print(train_df)

# 处理属性值 Fare
