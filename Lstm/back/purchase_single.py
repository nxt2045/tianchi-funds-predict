# - coding: utf-8 --
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 开始计时
start_time = time.clock()
test_date = time.time()
test_path = "./test/" + str(test_date)
print(os.makedirs(test_path))

# 定义常量
train_type = "purchase_1403to1407"
time_step = 30  # 时间步
rnn_unit = 50  # hidden layer units
batch_size = 60  # 每一批次训练多少个样例
input_size = 1  # 输入层维度
output_size = 1  # 输出层维度
lr = 0.001  # 学习率
train_range = 500  # 重复训练次数

# ——————————————————导入数据——————————————————————
df = pd.read_csv(r"./data/processed/prt.csv", sep=',', engine='python',
                 encoding='utf-8',
                 parse_dates=['report_date'])

df.set_index(['report_date'], inplace=True)
df_1403to1407 = df['2014-03':'2014-07']
df_1403to1408 = df['2014-03':'2014-08']
df_1408 = df['2014-08']


data_8 = np.array(df_1408['purchase'])
print(data_8)

data_38 = np.array(df_1403to1408['purchase'])
normalize_total = (data_38 - np.mean(data_38)) / np.std(data_38)  # 标准化
normalize_total = normalize_total[:, np.newaxis]  # 增加维度

data_37 = np.array(df_1403to1407['purchase'])
normalize_data = (data_37 - np.mean(data_37)) / np.std(data_37)  # 标准化
normalize_data = normalize_data[:, np.newaxis]  # 增加维度

# 以折线图展示data
# plt.figure()
# plt.plot(data)
# plt.show()

# 生成训练集
train_x, train_y = [], []  # 训练集
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

# ——————————————————定义神经网络变量——————————————————
X = tf.placeholder(tf.float32, [None, time_step, input_size])  # 每批次输入网络的tensor
Y = tf.placeholder(tf.float32, [None, time_step, output_size])  # 每批次tensor对应的标签
# 输入层、输出层权重、偏置
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(batch):  # 参数：输入网络批次数目
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    pred, _ = lstm(batch_size)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练train_range次
        for i in range(train_range):
            step = 0
            start = 0
            end = start + batch_size
            while (end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size
                # 每10步保存一次参数
                if step % 10 == 0:
                    print("Number of iterations:", i, " loss:", loss_)
                    print("model_save", saver.save(sess, test_path + "/purchase_save1/model.ckpt"))
                step += 1


# ————————————————预测模型————————————————————
def prediction():
    pred, _ = lstm(1)  # 预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint(test_path + "/purchase_save1")
        saver.restore(sess, module_file)
        # 取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        prev_seq = train_x[-1]
        normalize_predict = []
        # 得到之后31个预测结果
        for i in range(31):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            normalize_predict.append(next_seq[-1])
            # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        # 结束计时
        end_time = time.clock()

        # 以折线图表示结果
        plt.figure(figsize=(35, 7))
        plt.plot(list(range(len(normalize_total))), normalize_total, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(normalize_predict))), normalize_predict,
                 color='r')
        plt.savefig(test_path + "/test_date=" + str(test_date) + "_single_" + train_type + "_ts=" + str(
            time_step) + "_bs=" + str(
            batch_size) + "_lr=" + str(
            lr) + "_" + str(train_range) + ".png")
        plt.show()

        # 去正则化
        predict = np.mean(data_37) + np.array(normalize_predict) * np.std(data_37)
        # 计算error
        purchase_error = np.divide(np.abs(data_8 - predict.flatten()), data_8)
        purchase_score = 0
        for item in purchase_error:
            print(item)
            if item <= 0.3:
                purchase_score += 10 - (100.0 / 3) * item
        print('purchase_score:\n', purchase_score)
        text_file = open("./test/purchase_score.txt", "a+")
        text_file.write(
            'test_date:%f %s(single) ts:%d bs:%d lr:%f train_range:%d time:%f score:%f' % (test_date, train_type,
                                                                                           time_step, batch_size, lr,
                                                                                           train_range,
                                                                                           end_time - start_time,
                                                                                           purchase_score))
        text_file.write('\n')
        text_file.close()

        # 导出结果
        index = pd.date_range(start='20140801', end='20140831')
        predict = pd.DataFrame(predict, index=index)
        predict.to_csv(
            test_path + "/test_date=" + str(test_date) + "_single_" + train_type + "_ts=" + str(
                time_step) + "_bs=" + str(batch_size) + "_lr=" + str(
                lr) + "_" + str(train_range) + ".csv", header=0)


with tf.variable_scope('train'):
    train_lstm()

with tf.variable_scope('train', reuse=True):
    prediction()
