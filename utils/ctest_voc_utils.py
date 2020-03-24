#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.tfrecord_voc_utils as voc_utils
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
output_path='D:/install/RetinaNet/RetinaNet-tensorflow/data/'
Annotations='E:/image/libin_dataset/RetinaNet/VOC/Annotations/'
JPEGImages='E:/image/libin_dataset/RetinaNet/VOC/JPEGImages/'
if os.path.exists(Annotations)==False:
    os.makedirs()

if os.path.exists(Annotations)==False:
    os.makedirs(JPEGImages)

tfrecord = voc_utils.dataset2tfrecord(Annotations, JPEGImages,
                                      output_path, 'test',2)
print(tfrecord)
