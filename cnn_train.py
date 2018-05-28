# -*- coding:utf-8 -*-
import tensorflow as tf
import sys
sys.path.append('/media/bao/panD/ForU/competition/TMD/code/')
import pre_process


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],strides=[1, 4, 4, 1], padding='SAME')

def max_pool_5x5(x):
  return tf.nn.max_pool(x, ksize=[1, 5, 5, 1],strides=[1, 5, 5, 1], padding='SAME')


img_train, label_train = pre_process.read_and_decode('/media/bao/panD/ForU/competition/TMD/data/train-text/img-tfrecord')
img_test, label_test = pre_process.read_and_decode('/media/bao/panD/ForU/competition/TMD/data/test-text/img-tfrecord')

x = tf.placeholder(tf.float32, shape=[None, 200, 200])
y_ = tf.placeholder(tf.int32, shape=[None, 100])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 200, 200, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_5x5(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_5x5(h_conv2)

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 100])
b_fc2 = bias_variable([100])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdagradOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.nn.in_top_k(y_conv, tf.a    rgmax(y_, 1), 5)
cast = tf.cast(correct_prediction, tf.int32)
accuracy = tf.reduce_mean(cast)

saver = tf.train.Saver(max_to_keep=300)

img_batch, label_batch = tf.train.shuffle_batch([img_train, label_train], batch_size=128, capacity=20000, min_after_dequeue=1000)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(20000):
        img_batch_val, label_batch_val = sess.run([img_batch, label_batch])
        label_batch_val = [[1 if j == label_val else 0 for j in range(100)] for label_val in label_batch_val]
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: img_batch_val, y_: label_batch_val, keep_prob: 1.0})
            loss = cross_entropy.eval(feed_dict={x: img_batch_val, y_: label_batch_val, keep_prob: 1.0})
            print("step %d, training accuracy %g, loss %g" % (i, train_accuracy, loss))
            # 保存模型参数，注意把这里改为自己的路径
            saver.save(sess, '/media/bao/panD/ForU/competition/TMD/data/model/modek0528/model.ckpt')
        # correct_prediction = correct_prediction.eval(feed_dict={x: img_batch_val, y_: label_batch_val, keep_prob: 1.0})
        # cast = cast.eval(feed_dict={x: img_batch_val, y_: label_batch_val, keep_prob: 1.0})
        train_step.run(feed_dict={x: img_batch_val, y_: label_batch_val, keep_prob: 0.5})
print("test accuracy %g" % accuracy.eval(feed_dict={x: img_test_val, y_: label_test_val, keep_prob: 1.0}))



