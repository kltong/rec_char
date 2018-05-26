#coding=utf-8
import os

import data_pre
import tensorflow as tf
import ToolsForCNN
import shutil
import os

# 数据准备,加载训练和测试数据
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# path = '/media/tf80226678/pan1/PyCharm/PycharmProjects/SentimentMonitor/files/trainclassfier/'
path = project_path+os.sep+'files'+os.sep+'trainclassfier'
(x_train,y_train,x_test,y_test) = data_pre.prepare(path)
print 'done'

# 输入输出定义
x = tf.placeholder(tf.float32,[None,50,10])
y_ = tf.placeholder(tf.float32,[None,2])

# 第一层卷积池化
x_image = tf.reshape(x,[-1,50,10,1])
W_conv1 = ToolsForCNN.weight_variable([5,10,1,8],"W_conv1")
b_conv1 = ToolsForCNN.bias_variables([8],"b_conv1")
h_conv1 = ToolsForCNN.tf.nn.relu(ToolsForCNN.conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = ToolsForCNN.avg_pool_5x1(h_conv1)

# 第二层卷积池化
W_conv2 = ToolsForCNN.weight_variable([5,10,8,16],"W_conv2")
b_conv2 = ToolsForCNN.bias_variables([16],"b_conv2")
h_conv2 = tf.nn.relu(ToolsForCNN.conv2d1(h_pool1,W_conv2)+b_conv2)
h_pool2 = ToolsForCNN.avg_pool_5x1(h_conv2)

# 全连接层
W_fc1 = ToolsForCNN.weight_variable([2*1*16,80],"W_fc1")
b_fc1 = ToolsForCNN.bias_variables([80],"b_fc1")
h_pool2_reshape = tf.reshape(h_pool2,[-1,2*1*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_reshape,W_fc1)+b_fc1)

# 输出层
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = ToolsForCNN.weight_variable([80,2],"W_fc2")
b_fc2 = ToolsForCNN.bias_variables([2],"b_fc2")
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# 损失函数定义和训练参数设置
cross_entyopy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
train_step = tf.train.AdagradOptimizer(1e-3).minimize(cross_entyopy)
correct_prediction = tf.equal(tf.arg_max(y_conv,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
# tf.global_variables_initializer()
saver = tf.train.Saver()
# model_path = '/media/tf80226678/pan1/PyCharm/PycharmProjects/SentimentMonitor/files/model/0202-01/'
model_path = project_path+os.sep+'files'+os.sep+'model'+os.sep+'0202-01' '/media/tf80226678/pan1/PyCharm/PycharmProjects/SentimentMonitor/files/model/0202-01/'
# 开始训练过程
accuracy_register = 0
accuracy_max = 0
for i in range(1000000):
    # 每一轮训练从训练集中挑选50条样本作为训练数据
    # (genDataSetArray, genLabelSetArray) = data_pre.seletrainset(x_train, y_train,30)
    (genDataSetArray, genLabelSetArray) = data_pre.generateOneStepTrianSet(x_train, y_train)
    # 按照一定的频度打印当前模型准确率
    if i%100 == 0:
        # train_accuracy = accuracy.eval(feed_dict = {x:genDataSetArray,y_:genLabelSetArray,keep_prob:1.0})
        print "step %d,training accuracy %g"%(i,accuracy.eval(feed_dict={x: genDataSetArray, y_: genLabelSetArray, keep_prob: 1.0}))
        accuracy_register = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
        print 'test accruacy:'+str(accuracy_register)
        if accuracy_register>accuracy_max:
            accuracy_max = accuracy_register
            shutil.rmtree(model_path)
            os.mkdir(model_path)
            model_file = model_path + 'sa_model.ckpt'
            save_path = saver.save(sess, model_file)
            print model_path+':'+str(accuracy_max)
        print cross_entyopy.eval(feed_dict={x: genDataSetArray, y_: genLabelSetArray, keep_prob: 1.0})
    # if i<100:
    # print i
    train_step.run(feed_dict={x:genDataSetArray,y_:genLabelSetArray,keep_prob:0.5})
print "test accuracy %g"%accuracy.eval(feed_dict={x:x_test,y_:y_test,keep_prob:1.0})

print len()
print len()

# print len(labels_test)
# print len(labels_train)
# print "------------------------"
# # print test_labelSetArray
# # 统计最终模型的性能指标，并输出错误分类的短信
# y_output = y_conv.eval(feed_dict={x:vecs_test,y_:labels_test,keep_prob:1.0})
# # for result in y_output:
# #     print result
# # pass
# for i in range(len(y_output)):
#     if y_output[i][0]>y_output[i][1]:
#         y_output[i] = [1.0,0.0]
#     else:
#         y_output[i] = [0.0,1.0]
#     pass
# P = 0
# N = 0
# FP = 0
# FN = 0
# fpIndex = []
# fnIndex = []
# for i in range(len(labels_test)):
#     if labels_test[i][0]==1.0:
#         P += 1
#         if y_output[i][1]==1.0:
#             FN += 1
#             fnIndex.append(vecs_test[i])
#             # fnIndex.append(i)
#         pass
#     else:
#         N += 1
#         if y_output[i][0]==1.0:
#             FP += 1
#             fpIndex.append(vecs_test[i])
#             # fpIndex.append(i)
#         pass
#     pass
# save_path = saver.save(sess,"files/model/fraud_bankNormal_model")
# print "P : "+str(P)
# print "N : "+str(N)
# print "FP : "+str(FP)
# print "FN : "+str(FN)
# fpIndex = FormatSmsVec.vec2Sms(fpIndex)
# for sms in fpIndex:
#     print sms
# pass
# print "---------------------------------------------------------------------------------------------"
# fnIndex = FormatSmsVec.vec2Sms(fnIndex)
# for sms in fnIndex:
#     print sms
# pass
# print "完成！"