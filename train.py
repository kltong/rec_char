# coding=utf-8
import os

import tensorflow as tf
import sys
sys.path.append('/media/bao/panD/ForU/competition/TMD/code')
import pre_process

"""
说明: 
"""

weight1_row = 4
weight1_column = 4
weight1_out_channel = 4
weight2_row = 5
weight2_column = 5
weight2_out_channel = 8
weight2_in_channel = 4
weight3_row = 5
weight3_column = 5
weight3_out_channel = 16
weight3_in_channel = 8

weight_full_row = 256
weight_full_column = 100



x = tf.placeholder(tf.float32, [None, 200, 200])
y_ = tf.placeholder(tf.float32, [None, 100])

x_image = tf.reshape(x, [-1, 200, 200, 1])
weight1 = tf.Variable(tf.truncated_normal([weight1_row, weight1_column, 1, weight1_out_channel],
                                          stddev=0.1), name='weight1')
bias1 = tf.Variable(tf.zeros(weight1_out_channel), name='bias1')

weight2 = tf.Variable(tf.truncated_normal([weight2_row, weight2_column, weight2_in_channel, weight2_out_channel],
                                          stddev=0.1), name='weight2')
bias2 = tf.Variable(tf.zeros(weight2_out_channel), name='bias2')

weight3 = tf.Variable(tf.truncated_normal([weight3_row, weight3_column, weight3_in_channel, weight3_out_channel],
                                          stddev=0.1), name='weight1')
bias3 = tf.Variable(tf.zeros(weight3_out_channel), name='bias3')

weight_full = tf.Variable(tf.truncated_normal([weight_full_row, weight_full_column], stddev=0.1), name='weight_full')
bias_full = tf.Variable(tf.zeros(weight_full_column), name='bias_full')

conv1 = tf.nn.relu(tf.nn.conv2d(x_image, weight1, strides=[1, 1, 1, 1], padding='SAME') + bias1)
pool1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weight2, strides=[1, 1, 1, 1], padding='SAME') + bias2)
pool2 = tf.nn.avg_pool(conv2, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')

conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weight3, strides=[1, 1, 1, 1], padding='SAME') + bias3)
pool3 = tf.nn.avg_pool(conv3, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')

pool3_re = tf.reshape(pool3, [-1, 256])
y = tf.nn.softmax(tf.matmul(pool3_re, weight_full)+bias_full)

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y, tf.argmax(y_, 1), 5), "float"))
saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


path = '/media/bao/panD/ForU/competition/TMD/data/train'
images = []
images_labels = []
ch_char_dict = {}
pre_process.get_files_labels(path, images, images_labels, ch_char_dict, 'root_label')
x_train_list, y_train_list, x_test_list, y_test_list = pre_process.sep_test_train(images, images_labels)
# x_test_list = pre_process.get_pixel(x_test_list)

for i in range(30000):
    print('fffffffffffffffffffffffff')
    x_batch, y_batch = pre_process.next_batch(x_train_list, y_train_list, ch_char_dict, 100)
    # if i % 100 == 0:  # 训练100次，验证一次
    print 'jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj'
    train_acc = accuracy.eval(feed_dict={x: x_batch, y_: y_batch})
    print 'step %d, training accuracy %g' % (i, train_acc)
    if i % 10 == 0:
        saver.save(sess, path+os.path.sep+'model/%d'+'model.ckpt')
    train_step.run(feed_dict={x: x_batch, y_: y_batch})

test_acc = accuracy.eval(feed_dict={x: x_test_list,  y_: y_test_list})
print "test accuracy %g" % test_acc

