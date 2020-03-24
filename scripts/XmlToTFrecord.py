#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : XmlToTFrecord.py
# @Author: LiBin
# @Date  : 2019/12/13
# @Desc  :
import xml.etree.ElementTree as ET
import numpy as np
import os
import tensorflow as tf

from PIL import Image
import cv2

classes = ["划伤", "白边", "亮边"]

f = open('train.txt', 'a')
def roi_return(x,y,w,h,width,height):
    num_rows = int(height/800)
    num_cols = int(width/800)
    # for num_row in range(num_rows):
    #     for num_col in range(num_cols):
    center_x = x + int(w / 2)
    center_y = y + int(h / 2)
    num_col = int(center_x/800)
    num_row = int(center_y/800)
    a = center_x % 800
    b = center_y % 800
    # print(x, y, center_x, center_y, num_col, num_row, a, b)
    if num_col < num_cols and num_row < num_rows:
        point_x = num_col * 800
        point_y = num_row * 800
    if num_col == num_cols and num_row < num_rows:
        point_x = width - 800
        point_y = num_row * 800
    if num_col < num_cols and num_row == num_rows:
        point_x = num_col * 800
        point_y = height - 800
    if num_col == num_cols and num_row == num_rows:
        point_x = width - 800
        point_y = height - 800
    # print(point_x,point_y)
    xmin = x-point_x if (x-point_x)>0 else 0
    ymin = y-point_y if (y-point_y)>0 else 0
    xmax = x-point_x+w if (x-point_x+w)<800 else 800
    ymax = y-point_y+h if (y-point_y+h)<800 else 800
    return point_x, point_y, xmin, ymin, xmax, ymax


def convert_annotation(data_path,save_path):

    xml_ind_files = glob.glob(os.path.join(data_path,'*.xml'))
    for xml_ing in xml_ind_files:
        xml_name = os.path.basename(xml_ing)
        base_name = xml_name.split('.')[0]
        img_name = data_path+base_name+'.bmp'
        root = ET.parse(xml_ing).getroot()
        objects = root.findall('object')
        i = 0
        print(xml_ing)
        for obj in objects:
            i = i+1
            annotation = '/home/test/data/under_train/'+ base_name + '_' +str(i)+'.bmp'
            bbox = obj.find('bndbox')
            class_ind = classes.index(obj.find('name').text.lower().strip())
            xmin = bbox.find('xmin').text.strip()
            xmax = bbox.find('xmax').text.strip()
            ymin = bbox.find('ymin').text.strip()
            ymax = bbox.find('ymax').text.strip()

            x = int(xmin)
            y = int(ymin)
            w = int(xmax) - x
            h = int(ymax) - y

            img = cv2.imread(img_name)
            height,width,_ = img.shape
            point_x, point_y, xxmin, yymin, xxmax, yymax = roi_return(x, y, w, h,width,height)
            small_img = img[point_y:point_y+800,point_x:point_x+800,:]
            cv2.imwrite(save_path+base_name+'_'+str(i)+'.bmp',small_img)
            annotation += ' ' + ','.join([str(xxmin), str(yymin), str(xxmax), str(yymax), str(class_ind)])
            f.write(annotation + "\n")



def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]
xmlpath='E:/image/train_flat_dataset/only_have_xml_image/'

def convert_annotation(xmlfilepath,save_path):
    ################
    bboxes=[]
    xml_name = os.path.basename(xmlfilepath)
    base_name = xml_name.split('.')[0]
    img_name = xmlpath + base_name + '.bmp'
    root = ET.parse(xmlfilepath).getroot()
    objects = root.findall('object')
    i = 0
    print(xmlfilepath)
    for obj in objects:
        i = i + 1
        bbox = obj.find('bndbox')
        class_ind = classes.index(obj.find('name').text.lower().strip())
        xmin = bbox.find('xmin').text.strip()
        xmax = bbox.find('xmax').text.strip()
        ymin = bbox.find('ymin').text.strip()
        ymax = bbox.find('ymax').text.strip()

        x = int(xmin)
        y = int(ymin)
        w = int(xmax) - x
        h = int(ymax) - y
        img = cv2.imread(img_name)
        height, width, _ = img.shape
        point_x, point_y, xxmin, yymin, xxmax, yymax = roi_return(x, y, w, h, width, height)
        small_img = img[point_y:point_y + 800, point_x:point_x + 800, :]
        bb = [xxmin, yymin, xxmax, yymax] + [class_ind]

        bboxes.extend(bb)

        # if len(bboxes) < 30 * 5:
        #     bboxes = bboxes + [0, 0, 0, 0, 0] * (30 - int(len(bboxes) / 5))

        write_2_loc=save_path + base_name + '_' + str(i) + '.bmp'
        print('write path:',write_2_loc)
        cv2.imwrite(write_2_loc, small_img)

        xywhc=np.array(bboxes, dtype=np.float32).flatten().tolist()
        img_raw = convert_img(write_2_loc)
        example = tf.train.Example(features=tf.train.Features(feature={
            'xywhc':
                tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
            'img':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))


        writer.write(example.SerializeToString())
    return np.array(bboxes, dtype=np.float32).flatten().tolist()






    ################


    # in_file = open(image_name,'r',encoding='UTF-8')
    #
    # tree = ET.parse(in_file)
    # root = tree.getroot()
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    # bboxes = []
    # for i, obj in enumerate(root.iter('object')):
    #
    #     difficult = obj.find('difficult').text
    #     cls = obj.find('name').text
    #     if cls not in classes or int(difficult) == 1:
    #         continue
    #     cls_id = classes.index(cls)
    #     xmlbox = obj.find('bndbox')
    #     b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
    #          float(xmlbox.find('ymax').text))
    #     bb = convert((w, h), b) + [cls_id]
    #
    #
    #     bboxes.extend(bb)
    # # if len(bboxes) < 30 * 5:
    # #     bboxes = bboxes + [0, 0, 0, 0, 0] * (30 - int(len(bboxes) / 5))
    #
    # return np.array(bboxes, dtype=np.float32).flatten().tolist()


def convert_img(bmp_name):
    image = Image.open(bmp_name)

    print(len(image.split()))
    resized_image = image.resize((500, 500), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    img_raw = image_data.tobytes()
    return img_raw


filename = os.path.join('E:/image/train_flat_dataset/only_have_xml_image/test'+ '.tfrecords')
writer = tf.python_io.TFRecordWriter(filename)

import glob

save_img='E:/image/train_flat_dataset/net/'
image_list = glob.glob(os.path.join(xmlpath,'*.xml'))

for xml_path in image_list:
    print(xml_path)

    xywhc = convert_annotation(xml_path,save_img)
    # bmp_name=xml_path[0:-4]+'.bmp'
    # img_raw = convert_img(bmp_name)
    #
    # example = tf.train.Example(features=tf.train.Features(feature={
    #     'xywhc':
    #         tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
    #     'img':
    #         tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    # }))
    #
    # writer.write(example.SerializeToString())
writer.close()