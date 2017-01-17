#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2017/1/15 19:20
@Author  : cai

Kaggle竞赛练习--Digit Recognition代码
"""
import csv
import operator
from numpy import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import time
from sklearn.ensemble import RandomForestClassifier

def toInt(array):
    """
    将字符串转换为整数
    :param array:
    :return:
    """
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m, n))
    for i in range(m):
        for j in range(n):
                newArray[i, j] = int(array[i, j])
    return newArray

def nomalizing(array):
    """
    归一化数据
    :param array:
    :return:
    """
    m,n=shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def loadTrainData(data):
    """
    载入训练数据，并返回训练数据和标签

    :param data: 训练数据文件，csv格式保存
    :return:
    """
    datas = []
    with open(data) as file:
        lines = csv.reader(file)
        for line in lines:
            # 42001 * 785
            datas.append(line)
    # 第一行是描述信息，可以去掉
    datas.remove(datas[0])
    datas = array(datas)
    label = datas[:, 0]
    data_ = datas[:, 1:]
    return nomalizing(toInt(data_)), toInt(label)

def loadTestData(testFile):
    """
    载入测试数据
    :param testFile:
    :return:
    """
    l = []
    with open(testFile) as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    # 28001*784
    l.remove(l[0])
    data = array(l)
    return nomalizing(toInt(data))

def loadTestResult(filename):
    """
    载入测试结果，即测试数据的预测值
    :param filename:
    :return:
    """
    l = []
    with open(filename) as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    # 28001*2
    l.remove(l[0])
    label = array(l)
    return toInt(label[:, 1])

def saveResult(result,csvName):
    """
    将预测结果保存到一个csv格式的文件中
    :param result: 结果列表
    :param csvName: 存放结果的csv文件名
    :return:
    """
    with open(csvName, 'w', newline='') as myFile:
        myWriter=csv.writer(myFile)
        tmp = ['ImageId', 'Label']
        myWriter.writerow(tmp)
        count = 1
        for i in result:
            tmp = []
            tmp.append(count)
            count += 1
            tmp.append(i)
            myWriter.writerow(tmp)

def classify_knn(inX, dataSet, labels, k):
    """
    knn算法
    :param inX: 输入的单个样本
    :param dataSet: 整个训练样本
    :param labels: 训练样本对应的标签
    :param k: knn算法选定的k值
    :return: 返回测试样本的预测标签
    """
    inX = mat(inX)
    dataSet = mat(dataSet)
    labels = mat(labels)
    # 获取dataSet的行数，也就是训练样本个数
    dataSetSize = dataSet.shape[0]
    # tile(A, (m,n)) 将数组A作为元素构造m行n列的数组
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = array(diffMat)**2
    # array.sum(axis=1)按行累加，axis=0为按列累加
    sqDistances = sqDiffMat.sum(axis=1)
    # 计算得到测试样本和每个训练样本的距离
    distances = sqDistances**0.5
    # array.argsort()，得到每个元素的排序序号
    # sortedDistIndicies[0]表示排序后排在第一个的那个数在原来数组中的下标
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[0, sortedDistIndicies[i]]
        # get(key,x)从字典中获取key对应的value，没有key的话返回0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sorted()函数，按照第二个元素即value的次序逆向（reverse=True）排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def knnClassify(trainData,trainLabel,testData):
    """
    调用scikit的knn算法包
    :param trainData:
    :param trainLabel:
    :param testData:
    :return:
    """
    # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf = KNeighborsClassifier()
    knnClf.fit(trainData, ravel(trainLabel))
    testLabel = knnClf.predict(testData)
    saveResult(testLabel, 'sklearn_knn_Result.csv')
    return testLabel

def svcClassify(trainData,trainLabel,testData):
    """
    调用sklearn的svm包
    :param trainData:
    :param trainLabel:
    :param testData:
    :return:
    """
    # default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    svcClf = svm.SVC(C=5.0)
    svcClf.fit(trainData, ravel(trainLabel))
    testLabel = svcClf.predict(testData)
    saveResult(testLabel, 'sklearn_svm_Result.csv')
    return testLabel

def random_forest_classifier(trainData,trainLabel,testData):
    """
    使用sklearn的 Random Forest
    """
    model = RandomForestClassifier(n_estimators=8)
    model.fit(trainData, trainLabel)
    testLabel = model.predict(testData)
    saveResult(testLabel, 'sklearn_rf_result.csv')
    return testLabel

def handwritingClassTest(trainFile, testFile, resultFile):
    """
    主函数，对测试数据进行预测并输出最终结果
    :param trainFile:
    :param testFile:
    :param resultFile:
    :return:
    """
    print("prepare trainData...")
    trainData, trainLabel = loadTrainData(trainFile)
    print("prepare testData...")
    testData = loadTestData(testFile)
    testLabel = loadTestResult(resultFile)
    m, n = shape(testData)
    errorCount = 0
    errorCount_svm = 0
    errorCount_rf = 0

    print("start to classify...")
    # 使用不同算法
    start_knn_time = time.time()
    print("using KNN...")
    # result1 = knnClassify(trainData, trainLabel, testData)
    print("totally taking %fs time" % (time.time() - start_knn_time))
    start_svm_time = time.time()
    print("using SVM...")
    # result2 = svcClassify(trainData, trainLabel, testData)
    print("totally taking %fs time" % (time.time() - start_svm_time))
    start_rf_time = time.time()
    print("using RF...")
    trainLabel = tile(trainLabel, (m, 1))
    result3 = random_forest_classifier(trainData, trainLabel, testData)
    print("totally taking %fs time" % (time.time() - start_rf_time))

    # 对比测试结果，输出错误率
    for i in range(m):
        if result1[i] != testLabel[0, i]:
            errorCount += 1
        if result2[i] != testLabel[0, i]:
            errorCount_svm += 1
        if result3[i] != testLabel[0, i]:
            errorCount_rf += 1
    print("\nthe total number of KNN errors is: %d, SVM error is: %d, RF error is: %d" % (errorCount, errorCount_svm, errorCount_rf))
    print("\nthe total KNN error rate is: %f, SVM error rate is: %f, RF error rate is: %f" % (errorCount/float(m), errorCount_svm/float(m), errorCount_rf/float(m)))

    # for i in range(m):
    #      classifierResult = classify_knn(testData[i], trainData, trainLabel, 5)
    #      resultList.append(classifierResult)
    #      print("the classifier %d came back with: %d, the real answer is: %d" % (i, classifierResult, testLabel[0, i]))
    #      if (classifierResult != testLabel[0, i]):
    #          errorCount += 1.0
    # print("\nthe total number of errors is: %d" % errorCount)
    # print("\nthe total error rate is: %f" % (errorCount/float(m)))
    # saveResult(resultList, "knn_result.csv")

trainFile = "train.csv"
testFile = "test.csv"
resultFile = "rf_benchmark.csv"
handwritingClassTest(trainFile, testFile, resultFile)
