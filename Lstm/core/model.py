# coding=gbk
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf


class Model:
    def __init__(self, save_path,test_date,data_path, col_start, col_end, train_begin, train_end,time_step,rnn_unit,lr,run_times):

        f = open(data_path)
        df = pd.read_csv(f)
        self.test_path =save_path +str(test_date)
        self.data = df.iloc[:, col_start:col_end].values  # 取第2-4列
        self.cols_num = (self.data).shape[1] - 1
        self.input_size = self.cols_num
        self.output_size = 1
        self.train_begin = train_begin
        self.train_end = train_end
        self.time_step = time_step
        self.rnn_unit = rnn_unit
        self.lr = lr
        self.run_times =run_times
        self.weights = {
            'in': tf.Variable(tf.random_normal([self.input_size, rnn_unit])),
            'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
        }
        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }


    def get_test_result(self):
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
        with tf.variable_scope('train'):
            self.train_lstm()
        with tf.variable_scope('train', reuse=True):
            test_y, test_predict = self.prediction()
            return test_y, test_predict



    # 获取训练集
    def get_train_data(self, batch_size=60, time_step=20):
        train_begin = self.train_begin
        train_end = self.train_end
        batch_index = []
        data_train = self.data[train_begin:train_end]
        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
        train_x, train_y = [], []  # 训练集
        for i in range(len(normalized_train_data) - time_step):
            if i % batch_size == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i + time_step, :self.cols_num]
            y = normalized_train_data[i:i + time_step, self.cols_num, np.newaxis]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data) - time_step))
        return batch_index, train_x, train_y

    # 获取测试集
    def get_test_data(self,time_step=30):
        test_begin = self.train_end
        data_test = self.data[test_begin:]
        print("get_test_data data_test:", len(data_test))
        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)
        normalized_test_data = (data_test - mean) / std  # 标准化
        print("get_test_data normalized_test_data:", len(normalized_test_data))
        size = (len(normalized_test_data) + time_step) // time_step  # 有size个sample
        # size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
        print("get_test_data size:", size)
        test_x, test_y = [], []
        for i in range(size - 1):
            x = normalized_test_data[i * time_step:(i + 1) * time_step, :self.cols_num]
            y = normalized_test_data[i * time_step:(i + 1) * time_step, self.cols_num]
            test_x.append(x.tolist())
            test_y.extend(y)
        test_x.append((normalized_test_data[(i + 1) * time_step:, :self.cols_num]).tolist())
        test_y.extend((normalized_test_data[(i + 1) * time_step:, self.cols_num]).tolist())
        return mean, std, test_x, test_y

    # ——————————————————定义神经网络变量——————————————————
    def lstm(self,X):
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        w_in = self.weights['in']
        b_in = self.biases['in']
        input = tf.reshape(X, [-1, self.input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, self.rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                     dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output = tf.reshape(output_rnn, [-1, self.rnn_unit])  # 作为输出层的输入
        w_out = self.weights['out']
        b_out = self.biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states

    # ——————————————————训练模型——————————————————
    def train_lstm(self,batch_size=80):
        time_step = self.time_step
        train_begin = self.train_begin
        train_end = self.train_end
        X = tf.placeholder(tf.float32, shape=[None, time_step, self.input_size])
        Y = tf.placeholder(tf.float32, shape=[None, time_step, self.output_size])
        batch_index, train_x, train_y = self.get_train_data(batch_size, time_step)
        pred, _ = self.lstm(X)
        # 损失函数
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        # module_file = tf.train.latest_checkpoint(checkpoint_dir="./")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, module_file)
            # 重复训练10000次
            for i in range(self.run_times):
                for step in range(len(batch_index) - 1):
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                   Y: train_y[batch_index[step]:batch_index[step + 1]]})
                print(i, loss_)
                if i % 200 == 0:
                    print("保存模型：", saver.save(sess, self.test_path + "/model_save/model.ckpt", global_step=i))



    # ————————————————预测模型————————————————————
    def prediction(self,time_step=30):
        X = tf.placeholder(tf.float32, shape=[None, time_step, self.input_size])
        # Y=tf.placeholder(tf.float32, shape=[None,time_step,self.output_size])
        mean, std, test_x, test_y = self.get_test_data(time_step)
        pred, _ = self.lstm(X)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint(checkpoint_dir=self.test_path + "/model_save")
            saver.restore(sess, module_file)
            test_predict = []
            for step in range(len(test_x) - 1):
                prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                predict = prob.reshape((-1))
                test_predict.extend(predict)
            test_y = np.array(test_y) * std[self.cols_num] + mean[self.cols_num]
            test_predict = np.array(test_predict) * std[self.cols_num] + mean[self.cols_num]
            acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差


            return test_y, test_predict




