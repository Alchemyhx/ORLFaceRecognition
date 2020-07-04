#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:26:39 2020
@author: Hexin Yuan 19210240055
"""
import glob
import numpy as np
import cv2


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


def min_distance(img_data, num, images):
    test_images = []
    for i in range(40):
        for j in range(5):
            test_images.append(img_data[10 * i + j])

    error_count_total = 0
    for i in range(num):
        error_count = 0
        print("以下为%d类样本识别结果：" % i)
        for j in range(5):
            t = img_data[i * 10 + 5 + j]
            min = 100000000
            min_num = -1
            for k in range(len(test_images)):
                distance = eucl_distance(test_images[k], t)
                if min > distance:
                    min = distance
                    min_num = int(k / 5)
            print(images[i * 10 + 5 + j], "：识别结果：", min_num)
            if min_num != i:
                error_count += 1
        error_count_total += error_count
        print("%d类错误个数：" % i, error_count, "，错误率：", error_count / 5)
    print("总错误个数：", error_count_total, "，总错误率", error_count_total / 200)


if __name__ == '__main__':
    X, images = import_data_from_orl("ORL/*.bmp")
    print(np.array(X).shape)
    # print(X[0])
    # print(X[1])
    min_distance(X, 40, images)