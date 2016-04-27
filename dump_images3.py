#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import sys
import warnings

filename_list = []

data_dir = './handwriting_chinese_100_classes'
dir_name = sys.argv[1]
files = [os.path.join(data_dir,dir_name,f) for f in os.listdir(os.path.join(data_dir,dir_name)) if os.path.isfile(os.path.join(data_dir,dir_name,f))]
filename_list.extend(files)

num_files = filename_list.__len__()
print >> sys.stderr, "number of files:%i"%(num_files)

images = []
labels = [dir_name]*num_files
images_tensor = tf.placeholder(tf.float32,[num_files,28*28])
for i in range(num_files):
 filename = filename_list[i]
 print >> sys.stderr, " %d,"%(i),
 filename_queue = tf.train.string_input_producer([filename])
 reader = tf.WholeFileReader()
 key, value = reader.read(filename_queue) #this reads only one file
 image = tf.image.decode_png(value)
 image = tf.image.resize_images(image, 28, 28)
 image = tf.to_float(image)*(1./255)-0.5
 image = tf.reshape(image,[1,28*28])
 image = tf.gather(image,0)
 images.append(image)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
 sess.run(init_op)

 #start populating the filename queue.
 coord = tf.train.Coordinator()
 threads = tf.train.start_queue_runners(sess=sess,coord=coord)

 images_tensor = tf.pack(images)
 images_tensor.eval(session=sess)
 
 f = open('dump%s.txt'%(dir_name), 'w')
 for i in range(num_files):
  f.write("%s\t"%(labels[i]))
  image = images[i].eval(session=sess)
  for j in range(image.__len__()):
   f.write("%5.4f "%(image[j]))
  f.write('\n')
  f.flush()
 f.close()

 coord.request_stop()
 coord.join(threads)
