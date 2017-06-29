#!/usr/local/python3

import tensorflow as tf

class FNN(object):
    def __init__(self, lr, drop_keep_rate, dim_input, dims_hid_layers=[], dim_output, \
        task_type='regression', L2_lambda=0.0):
        #rates
        self.lr = lr
        self.drop_keep_rate = drop_keep_rate

        #structure
        self.dim_input = dim_input
        self.dims_hid_layers = dims_hid_layers
        self.dim_output = dim_output

        #misc
        self.task_type = task_type
        self.L2_lambda = L2_lambda
        self.L2_penalty = tf.constant(0.0)

        #layers storage
        self.tensor_in = None
        self.tensors_hid = []
        self.tensor_out = None
        self.tensor_label = None

        #paras storage
        self.Hs = [] #output of all hidden layers
        self.Ws = [] #weights
        self.bs = [] #biases
        self.L2s = [] #L2s

        self.build_model('F')

    @staticmethod
    def summarize_variables(var, name):
        with tf.name_scope(name + '_summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope(name + '_stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('_stddev/' + name, stddev)
        tf.scalar_summary('_max/' + name, tf.reduce_max(var))
        tf.scalar_summary('_min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

    @staticmethod
    def init_weight(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def init_bias(shape):
        return tf.Variable(cf.constant(0.1, shape=shape))

    @staticmethod
    def decide_act(act_type)

        return act_out

    def through_layer(self, tensor_in, dim_in, dim_out, name_layer, act=tf.nn.relu):
        with tf.name_scope(name_layer):
            with tf.name_scope(name_layer + '_weights'):
                weights = init_weight([dim_in, dim_out])
                self.summarize_variables(weights, name_layer + '/weights')
                self.W.append(weights)
            with tf.name_scope(name_layer+'_biases'):
                biases = tf.Variable(tf.constant(0.1, [dim_out]))
                self.summarize_variables(biases, name_layer + '/biases')
            with tf.name_scope(name_layer+'_Wx_plus_b'):
                pre_activate = tf.matmul(tensor_in, weights) + biases
                tf.histogram_summary(name_layer + '/pre_activations', pre_activate)
            activations = act(pre_activate, name='activation')
            tf.histogram_summary(name_layer + '/activations', activations)
        return activations, tf.nn.l2_loss(weights)

    def dropout(self, in_tensor, drop_keep_rate):
        return tf.nn.dropout(in_tensor, drop_keep_rate)

    def build_model(self, prefix):
        #input
        with tf.name_scope('Input'):
            self.tensor_in = tf.placeholder(tf.float32, [None, dim_input], name="inputs")
        with tf.name_scope('Label'):
            self.tensor_label = tf.placeholder(tf.float32, [None, dim_label], name="labels")

        #hid layers
        tensor = self.tensor_in
        dims_layers = [self.dim_input] + self.dims_hid_layers + [self.dim_output]

        for i in range(len(self.dims_hid_layers)-1):
            tensor, L2_loss = self.through_hid_layer(tensor, dim_layers[i], dim_layers[i+1], \
                prefix+'_hid_'+str(i+1), act = tf.nn.relu)
            self.L2s.append(L2_loss)
            self.tensors_hid.append(tensor)
            tensor = self.dropout(tensor, drop_keep_rate)
            self.drops.append(tensor)
            print('Add dense layer: %sD --> %sD relu with drop_keep:%s' \
                %(dims_hid_layers[i], dims_hid_layers[i+1], drop_keep_rate))

        #output layer
        if self.task_type == 'softmax':
            act_out = tf.nn.softmax
        else
            act_out = tf.identity
        
        self.tensor_out, L2_loss = self.through_layer(tensor, dim_layers[-2], dim_layers[-1], \
            name_layer='output', act=out_act)
        self.L2s.append(L2_loss)
        print('Add output layer: linear  %sD --> %sD' %(dim_layers[-2], dim_layers[-1]))

        #L2
        with tf.name_scope('L2'):
            for L2 in self.L2s:
                self.L2_penalty += L2
            tf.scalar_summary('L2_penalty', self.L2_penalty)

        #Loss
        if self.task_type == 'softmax':
            entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
            with tf.name_scope('cross entropy'):
                self.loss = tf.reduce_mean(entropy)
                tf.scalar_summary('loss', self.loss)
            with tf.name_scope('accuracy'):
                correct_predictions = \
                tf.equal(tf.argmax(self.tensor_out, 1), tf.argmax(self.tensor_label, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.scalar_summary('accuracy', self.accuracy)
        else:
            with tf.name_scope('SSE'):
                loss = tf.reduce_mean((self.tensor_out - self.tensor_labels)**2)
                tf.scalar_summary('loss', loss)

        with tf.name_scope('loss'):
            self.total_loss = self.loss + self.L2_penalty * self.L2_lambda
            tf.scalar_summary('total_loss', self.total_loss)

        with tf.name_scope('train'):
            self.model = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)
