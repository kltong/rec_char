#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
"""
说明: 对建模之前图片进行预处理
"""

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import random

rows = 200
columns = 200


def get_files(root_path, files):
    """
    说明： 从传入空的list：files，递归将path路径下所有文件绝对路径列出到files
    """
    files_tmp = os.listdir(root_path)
    for eachFile in files_tmp:
        if os.path.isfile(root_path+eachFile):
            files.append(root_path + eachFile)
        else:
            get_files(root_path + os.path.sep + eachFile + os.path.sep, files)


def sep_test_train(input_list, output_list):
    train_input_list = []
    train_output_list = []
    test_input_list = []
    test_output_list = []
    for i in range(len(input_list)):
        if 0 == random.randint(0, 2) % 3:
            test_input_list.append(input_list[i])
            test_output_list.append(output_list[i])
        else:
            train_input_list.append(input_list[i])
            train_output_list.append(output_list[i])
    return train_input_list, train_output_list, test_input_list,test_output_list


def get_files_labels(root_path, files, labels, label_dict, now_label):
    """
    说明： 从传入空的list：files，递归将path路径下所有文件绝对路径列出到files
            labels 为标签列表
            now_label 为当前目录标签
    """
    index = 0
    files_tmp = os.listdir(root_path)
    for eachFile in files_tmp:
        if os.path.isfile(root_path+eachFile):
            files.append(root_path + eachFile)
            labels.append(now_label)
        else:
            get_files_labels(root_path + os.path.sep + eachFile + os.path.sep, files, labels, label_dict, eachFile)
            label_dict[eachFile] = index
            index += 1


def size_height(img_list):
    """
    说明：对传入的images列表中图片的高和宽进行统计显示
    """
    size_list = [list(Image.open(im).size) for im in img_list]
    m_list = [size[0] for size in size_list]
    n_list = [size[1] for size in size_list]
    plt.figure(12)
    plt.subplot(121)
    plt.title('height hist')
    plt.hist(m_list, bins=20, range=(0, 600), facecolor='green', edgecolor='blue', alpha=1, histtype='bar')
    plt.subplot(122)
    plt.title('weight hist')
    plt.hist(n_list, bins=20, range=(0, 600), facecolor='green', edgecolor='blue', alpha=1, histtype='bar')
    plt.show()


def get_pixel(image_path_list):
    """
    说明： 传入图片路径列表，返回列表中每张图片的像素值，
          返回列表每个元素都是np.array(200,200)
          图片读入后统一高宽位200x200
          传入不是列表则直接返回
    """
    if type(image_path_list) != list:
        return
    image_list = [np.asarray(Image.open(image_path).resize((rows, columns))).tolist() for image_path in image_path_list]
    return image_list


def next_batch(image_list, label_list, char_dict, batch_size):
    data_size = len(image_list)
    index_list = [random.randint(0, data_size-1) for q in range(batch_size)]

    x_batch = []
    for i in range(batch_size):
        x_batch.append(image_list[index_list[i]])

    y_batch = []
    for i in range(batch_size):
        y_signal = []
        label_index = char_dict[label_list[index_list[i]]]
        for j in range(100):
            y_signal.append(1 if j == label_index else 0)
        y_batch.append(y_signal)

    return get_pixel(x_batch), y_batch

    # return [image_list[index_list[i]] for i in range(batch_size)], \
    #        [[1 if j == char_dict[label_list[index_list[j]]] else 0 for i in range(100)] for j in range(batch_size)]


# path = '/media/bao/panD/ForU/competition/TMD/data/train'
# images = []
# images_labels = []
# ch_char_dict = {}
# get_files_labels(path, images, images_labels, ch_char_dict, 'root_label')
# x_train_list, y_train_list, x_test_list, y_test_list = sep_test_train(images, images_labels)
# x, y = next_batch(x_test_list, y_test_list, ch_char_dict, 100)

"""
说明: 测试像素矩阵与汉字对应关系
"""
# for i in range(len(x)):
#     img = Image.fromarray(x[i])
#     index_tmp = y[i].index(max(y[i]))
#     for key in ch_char_dict:
#         if ch_char_dict[key] == index_tmp:
#             img.show()
#             print key
"""
说明：测试label对应功能
"""
# ff = 0
# for i in range(len(y)):
#     index_tmp = y[i].index(max(y[i]))
#     for key in ch_char_dict:
#         if ch_char_dict[key] == index_tmp:
#             if key not in x[i]:
#                 ff += 1
# print ff

# print 'done1'
# # for char in ch_char_dict:
# #     print char+':'+str(ch_char_dict[char])
# # size_height(images)   # 通过分析高、宽分布直方图，取定缩放比例
# print time.time()
# images_pixel_list = get_pixel(images[0:100])
# print time.time()
# print 'done2'
# print time.time()

