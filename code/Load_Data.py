# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:27:03 2020

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:09:43 2019

@author: Lenovo
"""

from tensorflow.python.keras.utils import get_file  #用于从给定URL中下载文件的模块工具
import gzip                                         #用于解压的模块
import numpy as np                                  #用于数据处理的模块


def load_data():
    #创建列表，将下载的数据路径填入
    paths = [];
    paths.append(r"D:\study\dataset\train-labels-idx1-ubyte.gz")
    paths.append(r"D:\study\dataset\train-images-idx3-ubyte.gz")
    paths.append(r"D:\study\dataset\t10k-labels-idx1-ubyte.gz")
    paths.append(r"D:\study\dataset\t10k-images-idx3-ubyte.gz")
    
    #with as语句用于打开并读取文件操作
    #frombuffer将data以流的形式读入转化成ndarray对象
    #第一参数为stream,第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
    # gzip.open以rb（读文件）方式打开文件
    #reshape改变的数组的结构
    
    #读取训练数据标签到y_train
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    #读取训练数据（根据标签的个数以及图像尺寸将读取到的数组reshape成len*28*28，即len张28*28的图片数据
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    #返回图片和标签数据
    return (x_train, y_train), (x_test, y_test)   
