# -*- coding: utf-8 -*-
import cv2 as cv
import pickle as p
import math
import random
import numpy as np
import copy
import os

CROP_SIZE = 24

def load_CIFAR_part(filename):
  """ 载入cifar数据集的一个part """
  with open(filename, 'rb') as f:
    datadict = p.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ 载入cifar全部数据 """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d.bin' % (b,))
    X, Y = load_CIFAR_part(f)
    #Y = onehot_encoder(Y, nb_cls=10)
    xs.append(X)         #将所有batch整合起来
    ys.append(Y)
  Xtr = np.concatenate(xs) #使变成行向量,最终Xtr的尺寸为(50000,32,32,3)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_part(os.path.join(ROOT, 'test_batch.bin'))
  #Yte = onehot_encoder(Yte, nb_cls=10)
  return Xtr,Ytr, Xte, Yte

class Data_Generator():
    """数据生成器"""
    def __init__(self, Xtr, Ytr,):
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.len_dataset = Xtr.shape[0]


    def batch_gen(self, batch_size):
        """生成batch"""
        batch_per_epoch = math.ceil(self.len_dataset/batch_size)
        cnt_batch = 0
        while True:
            indices = list(range(self.len_dataset))
            if cnt_batch % batch_per_epoch == 0:
                random.shuffle(indices)
                idx = 0

            image = self.Xtr[idx:idx+batch_size]
            label = self.Ytr[idx:idx+batch_size]
            idx += batch_size

            cnt_batch += 1
            yield (image, label)

def random_crop(img):
    """随机剪裁到24*24"""
    h, w = img.shape[:2]
    nh, nw = CROP_SIZE, CROP_SIZE
    sx, sy = random.randint(0, w - nw), random.randint(0, h - nh)
    img = img[sy:sy+nh, sx:sx+nw]
    return img

def augmentation(image_batch):
    """数据增广"""
    post_image_batch = np.ones((image_batch.shape[0], CROP_SIZE, CROP_SIZE, image_batch.shape[3]))
    for i in range(image_batch.shape[0]):
        ori_img = image_batch[i]
        new_img = random_crop(ori_img)
        post_image_batch[i] = new_img
        #random flip
        #randon brightning
        #random contrast
        #standardization
    return post_image_batch

def onehot_encoder(label_batch, nb_cls):
    """独热编码"""
    batch_size = label_batch.shape[0]
    encoded_label_batch = np.zeros((batch_size,nb_cls), dtype='int8')
    for i in range(batch_size):
        cls = label_batch[i]
        encoded_label_batch[i][cls] = 1
    return encoded_label_batch
