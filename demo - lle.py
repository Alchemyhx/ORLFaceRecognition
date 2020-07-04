#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:22:05 2020
@author: Hexin Yuan 19210240055
"""
import numpy as np
import glob
import cv2
from sklearn import manifold
import matplotlib.pyplot as plt
import math


def eucl_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


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


def min_distance(train_images, test_images, images):
    error_count_total = 0
    for i in range(40):
        error_count = 0
        print("以下为%d类样本识别结果：" % i)
        for j in range(5):
            min = math.inf
            min_num = -1
            for k in range(len(train_images)):
                distance = eucl_distance(train_images[k], test_images[i * 5 + j])
                if min > distance:
                    min = distance
                    min_num = int(k / 5)
            print(images[i * 10 + 5 + j], "：识别结果：", min_num)
            if i != min_num:
                error_count += 1
        print("%d类错误个数：" % i, error_count, "，错误率：", error_count / 5)
        error_count_total += error_count
    print("总错误率为：", error_count_total / 200)

    return error_count_total


def lle(X, images, k, d):
    X_dim_d = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=d, method='standard').fit_transform(X)
    train_images = divide_train_images(X_dim_d)
    test_images = divide_test_images(X_dim_d)
    num = min_distance(train_images, test_images, images)


if __name__ == '__main__':
    X, images = import_data_from_orl("ORL/*.bmp")

    lle(X, images, 4, 40)

    train_images_2 = np.array(divide_train_images(X))
    X_dim_2 = manifold.LocallyLinearEmbedding(n_neighbors=4, n_components=2, method='standard').fit_transform(train_images_2)

    img_X = X_dim_2.T[0].T
    img_Y = X_dim_2.T[1].T

    fig = plt.figure()
    plt.scatter(img_X, img_Y, s=10)

    plt.plot(img_X[0], img_Y[0], marker='o', markerfacecolor='red')
    img = cv2.imread(images[0], 0)
    plt.figimage(img, xo=280, yo=100, cmap='gray')

    plt.plot(img_X[5], img_Y[5], marker='o', markerfacecolor='red')
    img = cv2.imread(images[10], 0)
    plt.figimage(img, xo=430, yo=230, cmap='gray')

    plt.plot(img_X[10], img_Y[10], marker='o', markerfacecolor='red')
    img = cv2.imread(images[20], 0)
    plt.figimage(img, xo=190, yo=280, cmap='gray')

    plt.show()
