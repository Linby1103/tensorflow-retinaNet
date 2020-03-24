# @File  : write_tfrecord.py
# @Author: LiBin
# @Date  : 2019/12/26
# @Desc  :


import tensorflow as tf
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 去'warning'
# shuffle_data = True
image_path ="C:/Users/EDZ/Desktop/new_dataset/"
# 取得该路径下所有图片的路径，type（addrs）= list
addrs = glob.glob(image_path)  # 标签数据的获得具体情况具体分析，type（labels）= list
txt_path = 'D:/workspace/code/cppdemo/yolov3/yolov3/yolov3/image_path.txt'
train_tfrecord_path = 'C:/Users/EDZ/Desktop/new_dataset/train.tfrecords'  # 输出文件地址


#读取标签txt文件
def readLabelsTxt(txt_path):
    imgName = []
    valueX = []
    with open(txt_path, 'r') as f:
        for line in f:
            key, value1 = line.split()
            imgName.append(key)
            valueX.append(value1)

#            print(imgCount, imgName[imgCount], valueX[imgCount], valueY[imgCount], valueR[imgCount])
    print('dataset samples num:', len(imgName))
    f.close()
    return imgName, valueX





def load_image(addr,input_size):  # A function to Load image
    img = cv2.imread(addr)
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# 将数据转化成对应的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# 把数据写入TFRecods文件
def create_tfrecord(imgName, posX):
    writer = tf.python_io.TFRecordWriter(train_tfrecord_path)  # 创建一个writer来写TFRecords文件
    for i in range(len(imgName)):
        img_path = imgName[i]
        print(i, ' Path:', img_path)
        #        img = cv2.imread(img_path)
        img = load_image(img_path,500)
        img = img.astype(np.uint8)
        img_raw = img.tostring()  # img.tobytes()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'img_raw': _bytes_feature(img_raw),
                'label': _int64_feature(int(posX[i])),

            }))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(is_train,train_tfrecord_path):
    filename_queue = tf.train.string_input_producer([train_tfrecord_path], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'img_posX': tf.FixedLenFeature([], tf.int64)
                                       })

    images = tf.decode_raw(features['img_raw'], tf.uint8)
    images = tf.reshape(images, [1200, 1600, 3])
    img_posX = features['img_posX']

    #    img_posX = tf.cast(features['img_posX'], tf.int32)
    #    img_posY = tf.cast(features['img_posY'], tf.int32)
    #    img_posR = tf.cast(features['img_posR'], tf.int32)
    if is_train == True:
        img_raw, labelX = tf.train.shuffle_batch([images, img_posX],
                                                                 batch_size=1,
                                                                 capacity=3,
                                                                 min_after_dequeue=3)
    else:
        img_raw, labelX = tf.train.batch([images, img_posX],
                                                         batch_size=1,
                                                         capacity=3)
    return img_raw, labelX,



if __name__ == '__main__':
    imgName, lable = readLabelsTxt(txt_path)
    create_tfrecord(imgName = imgName, posX = lable)
    image, lebel = read_and_decode(is_train=False,train_tfrecord_path=train_tfrecord_path)

    print('Read finish!')
    #
    # init_op = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     ## 启动多线程处理输入数据
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     a, b, c, d = sess.run([image, X, Y, R])
    #     print(b, c, d)
    #     coord.request_stop()
    #     coord.join(threads)
    #     aa = np.uint8(a[0, :, :, :])
    #     plt.imshow(aa)
    #     plt.show()