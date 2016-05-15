import tensorflow as tf
import numpy as np

"""Runs CNN for Chinese Characters."""
"""Dimension should be 784 and Num of classes should be 100"""

train_data = np.loadtxt("../dump_images/dump_train_data.txt")
train_label_file = open("../dump_images/dump_train_labels.txt")
train_labels = train_label_file.readlines(); train_label_file.close()

test_data = np.loadtxt("../dump_images/dump_test_data.txt")
test_label_file = open("../dump_images/dump_test_labels.txt")
test_labels = test_label_file.readlines(); test_label_file.close()

def labels_to_one_hot(labels_dense):
 """Convert class labels from scalars to one-hot vectors."""
 index = list(set(labels_dense))
 num_labels = len(index)
 num_instances = len(labels_dense)
 print"num_labels:%d num_instances:%d"%(num_labels,num_instances)
 labels_one_hot = np.zeros((num_instances,num_labels))
 for i in range(num_instances):
  labels_one_hot[i,index.index(labels_dense[i])] = 1
 #write index (index -> label) to a file
 index_file = open("../data/index.txt","w")
 for i in range(len(index)):
  index_file.write("%d:%s"%(i,index[i]))
 index_file.close()
 return labels_one_hot

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
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""From here, session begins"""
config = tf.ConfigProto(inter_op_parallelism_threads=2)
with tf.Session(config=config) as sess:
 sess.run(tf.initialize_all_variables())

 train_data = tf.convert_to_tensor(train_data, dtype=tf.float32).eval(session=sess)
 train_labels = labels_to_one_hot(train_labels)
 for i in range(10000):
  rand_rows = np.random.randint(len(train_data),size=50)
  batch_x = train_data[rand_rows,:]
  batch_y = train_labels[rand_rows,:]
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x: batch_x, y_: batch_y, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

 saver = tf.train.Saver()
 save_path = saver.save(sess, "../tmp/model.ckpt")
 print("Model saved in file: %s" % save_path)

 x_test = test_data
 y_test = labels_to_one_hot(test_labels)
 print("test accuracy %g"%accuracy.eval(feed_dict={
    x: x_test, y_: y_test, keep_prob: 1.0},session=sess))

