import tensorflow as tf
import numpy as np
import sys

filename_list = sys.argv[1:]
print filename_list

num_files = len(filename_list)
images = []
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

label_list = []
with open("../data/index.txt") as f:
 label_list = f.read().splitlines()
#print label_list

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 100])

W = tf.Variable(tf.zeros([784,100]))
b = tf.Variable(tf.zeros([100]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 100])
b_fc2 = bias_variable([100])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver()

config = tf.ConfigProto(inter_op_parallelism_threads=2)
with tf.Session(config=config) as sess:
 sess.run(tf.initialize_all_variables())
 #Restore variables from disk.
 saver.restore(sess, "../tmp/model.ckpt")

 for i in range(num_files):
  #start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)

  x_image_test = images[i].eval(session=sess)
  pred = np.argmax(y_conv.eval(feed_dict={
                  x: [x_image_test],
                  keep_prob: 1.0 })[0])
  print("pred:%d (%s)"%(pred,label_list[pred]))

  coord.request_stop()
  coord.join(threads)
