#!/usr/local/python3

import tensorflow as tf
import numpy as np

import time

class FNN(object):
    def __init__(self, lr, drop_keep_rate, dim_input, dims_hid_layers, dim_output, task_type='regression',
                 l2_lambda=0.0):
        #rates
        self.lr = lr
        self.drop_keep_rate = drop_keep_rate
        #structure
        self.dim_input = dim_input
        self.dims_hid_layers = dims_hid_layers
        self.dim_output = dim_output
        #misc
        self.task_type = task_type
        self.l2_lambda = l2_lambda
        self.l2_penalty = tf.constant(0.0)
        #layers storage
        self.tensor_in = None
        self.tensors_hid = []
        self.tensor_out = None
        self.tensor_label = None
        self.tensors_drop = []
        #paras storage
        self.Hs = [] #output of all hidden layers
        self.Ws = [] #weights
        self.bs = [] #biases
        self.L2s = [] #L2s
        #stats
        self.accuracy = 0.0
        self.loss = 0.0


        self.build_model('F')

    @staticmethod
    def summarize_variables(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('value', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('value', var)

    @staticmethod
    def init_weight(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def init_bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def through_layer(self, tensor_in, dim_in, dim_out, name_layer, act=tf.nn.relu):
        with tf.name_scope(name_layer):
            with tf.name_scope('weights'):
                weights = FNN.init_weight([dim_in, dim_out])
                self.summarize_variables(weights)
                self.Ws.append(weights)
            with tf.name_scope('biases'):
                biases = FNN.init_bias([dim_out])
                self.summarize_variables(biases)
            with tf.name_scope('Wx_plus_b'):
                pre_activate = tf.matmul(tensor_in, weights) + biases
                tf.summary.histogram('value', pre_activate)
            activations = act(pre_activate, name='activation')
            tf.summary.histogram('activations', activations)
        return activations, tf.nn.l2_loss(weights)

    @staticmethod
    def dropout(in_tensor, drop_keep_rate):
        return tf.nn.dropout(in_tensor, drop_keep_rate)

    def build_model(self, prefix):
        #input
        with tf.name_scope('Input'):
            self.tensor_in = tf.placeholder(tf.float32, [None, self.dim_input], name="inputs")
        with tf.name_scope('Label'):
            self.tensor_label = tf.placeholder(tf.float32, [None, self.dim_output], name="labels")

        #hid layers
        tensor = self.tensor_in
        dims_layers = [self.dim_input] + self.dims_hid_layers + [self.dim_output]

        for i in range(len(dims_layers)-2):
            tensor, L2_loss = self.through_layer(tensor, dims_layers[i], dims_layers[i+1], prefix+'_hid_'+str(i+1), act = tf.nn.relu)
            self.L2s.append(L2_loss)
            self.tensors_hid.append(tensor)
            tensor = self.dropout(tensor, self.drop_keep_rate)
            self.tensors_drop.append(tensor)
            print('Add dense layer: %sD --> %sD relu with drop_keep:%s' \
                %(dims_layers[i], dims_layers[i+1], self.drop_keep_rate))

        #output layer
        if self.task_type == 'softmax':
            act_out = tf.nn.softmax
        else:
            act_out = tf.identity

        self.tensor_out, L2_loss = self.through_layer(tensor, dims_layers[-2], dims_layers[-1],name_layer='output',
                                                      act=act_out)
        self.L2s.append(L2_loss)
        print('Add output layer: linear  %sD --> %sD' %(dims_layers[-2], dims_layers[-1]))

        #L2
        with tf.name_scope('L2'):
            for L2 in self.L2s:
                self.l2_penalty += L2
            tf.summary.scalar('L2_penalty', self.l2_penalty)

        #Loss
        if self.task_type == 'softmax':
            entropy = tf.nn.softmax_cross_entropy_with_logits(self.tensor_out, self.tensor_label)
            with tf.name_scope('cross entropy'):
                self.loss = tf.reduce_mean(entropy)
                tf.summary.scalar('loss', self.loss)
            with tf.name_scope('accuracy'):
                correct_predictions = \
                tf.equal(tf.argmax(self.tensor_out, 1), tf.argmax(self.tensor_label, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)
        else:
            with tf.name_scope('SSE'):
                self.loss = tf.reduce_mean((self.tensor_out - self.tensor_label)**2)
                tf.summary.scalar('loss', self.loss)

        with tf.name_scope('loss'):
            self.total_loss = self.loss + self.l2_penalty * self.l2_lambda
            tf.summary.scalar('total_loss', self.total_loss)

        with tf.name_scope('train'):
            self.model = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

def construct_dataset():
    inputs=[[0,0],[0,1],[1,0],[1,1]]
    outputs=[0,1,1,0]
    X=np.array(inputs).reshape((4,2)).astype('int16')
    Y=np.array(outputs).reshape((4,1)).astype('int16')

    return X, Y

def main():
    ff = FNN(lr=1e-3, drop_keep_rate = 1.0, dims_hid_layers=[2], dim_input=2, dim_output=1, task_type='regression',
             l2_lambda=1e-2)

    model, inputs, label, output = ff.model, ff.tensor_in, ff.tensor_label, ff.tensor_out
    X, Y = construct_dataset()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    merged = tf.summary.merge_all()

    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    train_writer = tf.summary.FileWriter('log'+'/train/'+ timestr, sess.graph)

    T = 5000
    feed_dict = {inputs: X, label: Y}
    for i in range(T):
        sess.run(model, feed_dict=feed_dict)
        if (i % 50) is 0:
            summary_str = sess.run(merged, feed_dict=feed_dict)
            train_writer.add_summary(summary_str, i)

if __name__ == '__main__':
    main()