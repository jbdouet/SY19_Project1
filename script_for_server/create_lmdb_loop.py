'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 256

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

train_lmdb = '/home/robotex/Documents/Data_Edouard/3classes_orig/train_lmdb/'
train_path = '/home/robotex/Documents/Data_Edouard/3classes_orig/'

list_dirs=[]
for subdir, dirs, files in os.walk(train_path):
	for dir in dirs :       
		list_dirs.append(os.path.join(subdir, dir))

list_files=[]
for subdir, dirs, files in os.walk(train_path):
	for file in files :       
		list_files.append(os.path.join(subdir, file))





train_data = [img for img in glob.glob("../input/train/*jpg")]
#test_data = [img for img in glob.glob("../input/test1/*jpg")]


print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(list_files):
	print img_path
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
	nb_dirs = len(list_dirs)
	test =0
	for i,name in enumerate(list_dirs):
		if str(name) in img_path:
			label=i
			test=1
			break
	if test==0:
		label= nb_dirs+1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()


