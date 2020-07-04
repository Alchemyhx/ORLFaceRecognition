#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:26:39 2020
@author: Hexin Yuan 19210240055
"""
from numpy import *
import numpy as np
import glob
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def import_data_from_orl(filename):
    images = glob.glob(filename)
    images.sort()
    X = []
    for img_name in images:
        img = cv2.imread(img_name, 0)
        data = np.asarray(img)
        data = np.reshape(data, 10304)
        X.append(data)
    return X, images


def eigen_face(FaceMat, selecthr=0.8):
    FaceMat = mat(FaceMat).T
    avgImg = mean(FaceMat, 1)
    diffTrain = FaceMat - avgImg
    eigvals, eigVects = linalg.eig(mat(diffTrain.T * diffTrain))
    eigSortIndex = argsort(-eigvals)
    for i in range(shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]] / eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:, eigSortIndex]
    return avgImg, covVects, diffTrain


def divide_test_images(img_data):
    test_images = []
    for i in range(40):
        for j in range(5):
            test_images.append(img_data[10 * i + 5 + j])
    return test_images


def divide_train_images(img_data):
    train_images = []
    for i in range(40):
        for j in range(5):
            train_images.append(img_data[10 * i + j])
    return train_images


if __name__ == '__main__':
    X, images = import_data_from_orl("ORL/*.bmp")
    train_images = divide_train_images(X)
    test_images = divide_test_images(X)
    avgImg, FaceVector, diffTrain = eigen_face(train_images, 0.9)

    label = [[i, i, i, i, i] for i in range(40)]
    label = reshape(label, (200, 1))
    te = FaceVector.T * diffTrain
    clf = LinearDiscriminantAnalysis()
    clf.fit_transform(te.T, label.ravel())
    tes = FaceVector.T * (mat(test_images[0]).T - avgImg)

    total_count = 0
    for i in range(40):
        count = 0
        print("以下为%d类样本识别结果：" % i)
        for j in range(5):
            temp = FaceVector.T * (mat(test_images[i * 5 + j]).T - avgImg)
            res = clf.predict(temp.T)
            print(images[i * 10 + 5 + j], "：识别结果：", int(res))
            if res == i:
                count += 1
        total_count += count
        print("%d类错误个数：" % i, 5 - count, "，错误率：", 1 - count / 5)
    print("总错误率为：", 1 - total_count / 200)

    img = np.reshape(avgImg, (112, 92))
    img = img.astype(np.uint8)
    cv2.imshow("avgImg", img)
    cv2.waitKey(0)
