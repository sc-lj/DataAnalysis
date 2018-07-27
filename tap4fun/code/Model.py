# coding:utf-8

import tensorflow as tf
try:
    from code.Script import *
    from code.Feature import *
except:
    from Script import *
    from Feature import *
from functools import reduce

from Chat_Warning import *
import os,math
import pandas as pd
import numpy as np

arg=argument()
log=log_config(__file__)

class CNN():
    def __init__(self,height):
        self.arg=arg
        self.height=height#
        self.input=tf.placeholder(dtype=tf.float32,shape=[None,height,varNum],name='input')
        self.target=tf.placeholder(dtype=tf.float32,shape=[None,1],name='label')
        self.keep_out=tf.placeholder(tf.float32)

        self.model()

    def mul(self,x,y):
        return x*y

    def weight(self,shape):
        return tf.Variable(tf.random_normal(shape,mean=0,stddev=1,dtype=tf.float32),name='weight')

    def biase(self,shape):
        return tf.Variable(tf.constant(0.,shape=shape,dtype=tf.float32),name='biase')

    def model(self,norm=False,on_train=True):
        # 第一层卷积
        input_x = tf.expand_dims(self.input, -1)
        filter_size=[self.arg.filter,self.arg.filter,1,self.arg.filter_num]
        Wx_plus_b=input_x
        if norm:  # 判断书否是 BN 层
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],  # 想要 normalize 的维度, [0] 代表 batch 维度
                # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
            )

            ema = tf.train.ExponentialMovingAverage(decay=0.5)  # exponential moving average 的 decay 度

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            # mean, var = mean_var_with_update()  # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var
            mean, var = tf.cond(on_train,  # on_train 的值是 True/False
                                mean_var_with_update,  # 如果是 True, 更新 mean/var
                                lambda: (  # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                                    ema.average(fc_mean),
                                    ema.average(fc_var)
                                )
                                )
            scale = tf.Variable(tf.ones([self.arg.filter_num]))
            shift = tf.Variable(tf.zeros([self.arg.filter_num]))
            epsilon = 0.001
            input_x = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
            # 上面那一步, 在做如下事情:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

        convd=tf.nn.conv2d(input_x,filter=self.weight(filter_size),strides=[1,1,2,1],padding='VALID',name='convd-1')
        biase_shape = [self.arg.filter_num]
        Wx_plus_b=tf.nn.bias_add(convd, self.biase(biase_shape))
        if norm:  # 判断书否是 BN 层
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],  # 想要 normalize 的维度, [0] 代表 batch 维度
                # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
            )
            scale = tf.Variable(tf.ones([self.arg.filter_num]))
            shift = tf.Variable(tf.zeros([self.arg.filter_num]))
            epsilon = 0.001
            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
            # 上面那一步, 在做如下事情:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

        activate=tf.nn.relu(Wx_plus_b)
        pool=tf.nn.max_pool(activate,ksize=[1,self.arg.filter,self.arg.filter,1],strides=[1,1,1,1],padding='VALID',name='pool-1')

        # 第二层卷积
        filter_size_1 = [self.arg.filter, self.arg.filter, self.arg.filter_num, self.arg.filter_num]
        convd = tf.nn.conv2d(pool, filter=self.weight(filter_size_1), strides=[1, 1, 1, 1], padding='VALID', name='convd-1')
        biase_shape = [self.arg.filter_num]
        Wx_plus_b=tf.nn.bias_add(convd, self.biase(biase_shape))
        if norm:  # 判断书否是 BN 层
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],  # 想要 normalize 的维度, [0] 代表 batch 维度
                # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
            )
            scale = tf.Variable(tf.ones([self.arg.filter_num]))
            shift = tf.Variable(tf.zeros([self.arg.filter_num]))
            epsilon = 0.001
            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
            # 上面那一步, 在做如下事情:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

        activate=tf.nn.relu(Wx_plus_b)
        pool1 = tf.nn.max_pool(activate, ksize=[1, self.arg.filter, self.arg.filter, 1], strides=[1, 1, 2, 1], padding='VALID',name='pool-1')

        # # 第三层卷积
        filter_size_2 = [self.arg.filter, self.arg.filter, self.arg.filter_num, self.arg.filter_num]
        convd = tf.nn.conv2d(pool1, filter=self.weight(filter_size_2), strides=[1, 1, 2, 1], padding='VALID', name='convd-1')
        biase_shape = [self.arg.filter_num]
        Wx_plus_b=tf.nn.bias_add(convd, self.biase(biase_shape))
        if norm:  # 判断书否是 BN 层
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],  # 想要 normalize 的维度, [0] 代表 batch 维度
                # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
            )
            scale = tf.Variable(tf.ones([self.arg.filter_num]))
            shift = tf.Variable(tf.zeros([self.arg.filter_num]))
            epsilon = 0.001
            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
            # 上面那一步, 在做如下事情:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

        activate=tf.nn.relu(Wx_plus_b)
        filter=activate.get_shape()[1]
        pool2 = tf.nn.max_pool(activate, ksize=[1,filter, self.arg.filter, 1], strides=[1, 1, 2, 1], padding='VALID',name='pool-1')

        shape=pool2.get_shape().as_list()[-3:]
        product= reduce(self.mul,shape)

        # 第一个全连接层
        til_vector=tf.reshape(pool2,shape=[-1,product])
        fcon1=tf.add(tf.matmul(til_vector,self.weight([product,10])),self.biase([10]))
        fcon1=tf.nn.relu(fcon1)
        fcon1=tf.nn.dropout(fcon1,self.keep_out)
        result1 = tf.nn.relu(fcon1)

        fcon2=tf.add(tf.matmul(result1,self.weight([10,1])),self.biase([1]))
        fcon2=tf.nn.relu(fcon2)
        fcon2=tf.nn.dropout(fcon2,self.keep_out)
        self.result = tf.nn.relu(fcon2)

        self.rmse_cost = tf.sqrt(tf.losses.mean_squared_error(self.result, self.target))


def train_model(cnn):
    with tf.Session() as sess:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=arg.lr).minimize(cnn.rmse_cost, global_step=global_step)

        loss_summary = tf.summary.scalar('loss', cnn.rmse_cost)

        out_dir=arg.out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        train_summary_dir = os.path.join(out_dir, 'summary', 'train')
        train_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_dir = os.path.join(out_dir, 'summary', 'dev')
        dev_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoints_dir = os.path.abspath(os.path.join(out_dir, 'model'))
        checkpoint_prefix = os.path.join(checkpoints_dir, 'model')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=arg.max_checkpoints)


        def train_step(X, Y):
            feed_dict = {cnn.input: X, cnn.target: Y,cnn.keep_out:arg.dropout}
            _, step, summary, loss = sess.run([optimizer, global_step, loss_summary, cnn.rmse_cost], feed_dict=feed_dict)
            # print('train step {},loss {:g}'.format( step, loss))
            log.info('train step {},loss {:g}'.format( step, loss))
            train_writer.add_summary(summary, step)

        def dev_step(X, Y):
            feed_dict = {cnn.input: X, cnn.target: Y,cnn.keep_out:1}
            step, summary, loss = sess.run([global_step, loss_summary, cnn.rmse_cost], feed_dict=feed_dict)
            # print('dev step {},loss {:g}'.format(step, loss))
            log.info('dev step {},loss {:g}'.format(step, loss))
            dev_writer.add_summary(summary, step)

        meta=tf.train.get_checkpoint_state(checkpoints_dir)
        # #判断模型是否存在
        if meta and meta.model_checkpoint_path:
            saver.restore(sess,meta.model_checkpoint_path)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)


        for matrix, label in gen_train(train_file, batch=50000,log=log):
            train_step(matrix, label)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % arg.evaluate_every== 0:
                dev_data = gen_batch(valid_file, batch=50000)
                mat, lab = dev_data.__next__()
                dev_step(mat, lab)
                path = saver.save(sess, save_path=checkpoint_prefix, global_step=current_step)
                log.info("Saved model checkpoint to {}\n".format(path))

def predict_model(cnn):
    out_dir = arg.out_dir
    checkpoints_dir = os.path.abspath(os.path.join(out_dir, 'model'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        meta=tf.train.get_checkpoint_state(checkpoints_dir)
        # #判断模型是否存在
        if meta and meta.model_checkpoint_path:
            saver.restore(sess,meta.model_checkpoint_path)
        else:
            raise FileNotFoundError("未保存任何模型")

        # saver = tf.train.import_meta_graph(checkpoints_dir+"/model-2300.meta")
        # saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))

        dev_data = gen_batch(tap_fun_test, batch=10000,predict=True)
        Y=[]
        while True:
            try:
                mat, _ = dev_data.__next__()
            except StopIteration:
                break
            y=sess.run([cnn.result],feed_dict={cnn.input:mat,cnn.keep_out:1})
            y=y[0].flatten().tolist()
            print(y)
            Y.extend(y)
        print(len(Y))
        data=pd.read_csv(valid_file,index_col=0)
        new_data=pd.DataFrame(Y,index=data.index,columns=[Depvar])
        new_data.to_csv('../data/sub.csv')



if __name__ == '__main__':
    filt=arg.filter
    varNum=106#自变量总个数
    # 总共需要106个变量，去掉了userid和register_time两个变量
    height=int(math.ceil(float(varNum)/filt))
    cnn=CNN(height)
    # try:
    #     train_model(cnn)
    # except Exception as e:
    #     send_msg('tap4fun cnn模型出现错误，%s'%e)
    # send_msg('tap4fun cnn模型已经训练完毕')

    predict_model(cnn)







