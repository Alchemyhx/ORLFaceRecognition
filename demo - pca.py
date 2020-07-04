#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:26:39 2020
@author: Hexin Yuan 19210240055
"""
import glob
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
    # X = np.array(X).squeeze().T
    return X, images


def min_distance(train_images, test_images):
    error_count_total = 0

    for i in range(len(test_images)):
        min = 100000000
        min_num = -1
        for j in range(len(train_images)):
            distance = eucl_distance(test_images[i], train_images[j])
            if min > distance:
                min = distance
                min_num = int(j / 5)
        if int(i / 5) != min_num:
            error_count_total += 1

    return error_count_total


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
    X = np.array(X)

    test_img_dim = [5, 20, 100, 200]
    for i in test_img_dim:
        pca = PCA(n_components=i)
        new_X = pca.fit_transform(X)
        new_X_img = pca.inverse_transform(new_X).astype(np.uint8)
        new_X_img_1 = np.reshape(new_X_img[0], (112, 92))
        cv2.imshow("test_img", new_X_img_1)
        cv2.waitKey(0)

    dim = []
    error_count_Y = []
    for i in range(5, 200, 5):
        dim.append(i)
        pca = PCA(n_components=i)
        new_X = pca.fit_transform(X)
        train_images = divide_train_images(new_X)
        test_images = divide_test_images(new_X)
        print(np.array(train_images).shape, np.array(test_images).shape)
        num = min_distance(train_images, test_images)
        error_count_Y.append(num / 200)
        new_X_1 = pca.inverse_transform(new_X)

    plt.title("pca")
    plt.xlabel("dimension", fontsize=14)
    plt.ylabel("error rate", fontsize=14)
    plt.plot(dim, error_count_Y)
    plt.show()
