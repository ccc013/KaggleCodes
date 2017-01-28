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

# 处理属性值 Fare, 填充缺失值
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)
# print(train_df.info())
# print('-' * 40)
# print(test_df.info())
# print(test_df['Fare'])

# 绘图
# Fare 直方图
train_df['Fare'].plot(kind='hist', figsize=(15, 3), bins=100, xlim=(0, 50))
# Fare和Survived关系的图表
fare_not_survived = train_df['Fare'][train_df['Survived'] == 0]
fare_survived = train_df['Fare'][train_df['Survived'] == 1]
fare_avg = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
std_fare.index.names = fare_avg.index.names = ['Survived']
fare_avg.plot(kind='bar', yerr=std_fare, legend=False)
# plt.show()

# 处理Age
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
axis1.set_title('Age original value')
axis2.set_title('New age values')
# age in train
train_age_avg = train_df['Age'].mean()
train_age_std = train_df['Age'].std()
train_count_age_nan = train_df['Age'].isnull().sum()
# age in test
test_age_avg = test_df['Age'].mean()
test_age_std = test_df['Age'].std()
test_count_age_nan = test_df['Age'].isnull().sum()

# generate new ages in range [mean-3*std, mean + 3*std]
rand_1 = np.random.randint(train_age_avg - 3 * train_age_std,
                           train_age_avg + 3 * train_age_std,
                           train_count_age_nan)
rand_2 = np.random.randint(test_age_avg - 3 * test_age_std,
                           test_age_avg + 3 * test_age_std,
                           test_count_age_nan)
# plot original age hist
train_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
# fill nan values
train_df['Age'][np.isnan(train_df['Age'])] = rand_1
test_df['Age'][np.isnan(test_df['Age'])] = rand_2
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)

# plot new age hist
train_df['Age'].hist(bins=70, ax=axis2)
# # kde plot of age vs survived
facet = sns.FacetGrid(train_df, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
age_avg = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
fig, axis = plt.subplots(1,1,figsize=(18,4))
sns.barplot(x='Age', y='Survived', data=age_avg, ax=axis)

# Carbin值由于有太多缺失值，所以选择丢弃该属性值
train_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)

# Family，对该属性值是将是否有父母子女和兄弟姐妹合成一个属性值，如果有就是1，没有就是0
train_df['Family'] = train_df["Parch"] + train_df["SibSp"]
train_df['Family'].loc[train_df['Family'] > 0] = 1
train_df['Family'].loc[train_df['Family'] == 0] = 0

test_df['Family'] = test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

train_df = train_df.drop(['SibSp', 'Parch'], axis=1)
test_df = test_df.drop(['SibSp', 'Parch'], axis=1)

# Sex，对于小于16岁的孩子有很大几率生存，所以将乘客分成男性，女性和小孩
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

train_df['Person'] = train_df[['Age', 'Sex']].apply(get_person, axis=1)
test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis=1)

# 抛弃Sex属性，因为创建新的属性值Person
train_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)

# 将Person属性值变成一维向量，并放弃male属性值，由于其有着最低的平均存活率
person_dummies_train = pd.get_dummies(train_df['Person'])
person_dummies_train.columns = ['Child', 'Female', 'Male']
person_dummies_train.drop(['Male'], axis=1, inplace=True)

person_dummies_test = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child', 'Female', 'Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train_df = train_df.join(person_dummies_train)
test_df = test_df.join(person_dummies_test)

train_df.drop('Person', axis=1, inplace=True)
test_df.drop('Person', axis=1, inplace=True)

# Pclass
sns.factorplot('Pclass', 'Survived', order=[1, 2, 3], data=train_df, size=5)
pclass_dummies_train = pd.get_dummies(train_df['Pclass'])
pclass_dummies_train.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train_df.drop('Pclass', axis=1, inplace=True)
test_df.drop('Pclass', axis=1, inplace=True)

train_df = train_df.join(pclass_dummies_train)
test_df = test_df.join(pclass_dummies_test)

# 显示最后得到的所有特征值
# print(train_df.info())

# 定义训练集和测试集
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df

# 开始使用机器学习算法进行训练和预测
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os

rootPath = 'E:\\KaggleCodes\\TitanicTrain'
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_lr = logreg.predict(X_test)
# save result
test_data = pd.read_csv('test.csv')
result = pd.DataFrame({'PassengerId': test_data['PassengerId'].as_matrix(), 'Survived':Y_pred_lr.astype(np.int32)})
resultPath = os.path.join(rootPath,"logistic_regression_predictions.csv")
result.to_csv(resultPath, index=False)
print('lr:', Y_pred_lr)
# print('LogisticRegression_score = ', logreg.score(X_train, Y_train))

# Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)
result = pd.DataFrame({'PassengerId': test_data['PassengerId'].as_matrix(), 'Survived': Y_pred_rf.astype(np.int32)})
resultPath = os.path.join(rootPath, "random_forest_predictions.csv")
result.to_csv(resultPath, index=False)
print('rf:', Y_pred_rf)
# print('rf_score = ', random_forest.score(X_train, Y_train))

# xgboost
# from sklearn.cross_validation import *
# from sklearn.grid_search import GridSearchCV
# xgb_model = xgb.XGBClassifier()
# parameters = {'nthread': [4], #when use hyperthread, xgboost may become slower
#               'objective':['binary:logistic'],
#               'learning_rate': [0.05, 0.1, 0.15], #so called `eta` value
#               'max_depth': [5, 8],
#               'min_child_weight': [3, 11],
#               'silent': [1],
#               'subsample': [0.9],
#               'colsample_bytree': [0.5],
#               'n_estimators': [100, 300, 500, 800], #number of trees
#               'seed': [1337]}
#
# # #evaluate with roc_auc_truncated
# # def _score_func(estimator, X, y):
# #     pred_probs = estimator.predict_proba(X)[:, 1]
# #     return roc_auc_truncated(y, pred_probs)
#
# #should evaluate by train_eval instead of the full dataset
# clf = GridSearchCV(xgb_model, parameters, n_jobs=4,
#                    cv=StratifiedKFold(Y_train, n_folds=5, shuffle=True),
#                    verbose=2, refit=True)
#
# clf.fit(X_train, Y_train)
# best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
# print('Raw AUC score:', score)


