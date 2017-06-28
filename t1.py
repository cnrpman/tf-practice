import tensorflow as tf
import numpy as np

#train paras
lr = 1e-3
T = 20000

#model paras
D_input = 2
D_hidden = 2
D_label = 1

def model_building(D_input=2, D_label=1, D_hidden=2):
	x = tf.placeholder(tf.float32, [None, D_input], name="x")
	t = tf.placeholder(tf.float32, [None, D_label], name="t")

	#<Input layer>

	W_h1 = tf.Variable(tf.random_normal([D_input, D_hidden], stddev=0.1), name="W_h")
	b_h1 = tf.Variable(tf.constant(0.1, shape=[D_hidden]), name="b_h")
	pre_act_h1 = tf.matmul(x, W_h1) + b_h1
	act_h1 = tf.nn.relu(pre_act_h1, name='act_h')

	#<Hidden layer 1>
	 
	W_o = tf.Variable(tf.random_normal([D_hidden, D_label], stddev=0.1), name="W_o")
	b_o = tf.Variable(tf.constant(0.1, shape=[D_label]), name="b_o")
	pre_act_o = tf.matmul(act_h1, W_o) + b_o
	y = tf.nn.relu(pre_act_o, name='act_Y')

	#<Output layer>

	loss = tf.reduce_mean((y - t) ** 2)
	model = tf.train.AdamOptimizer(lr).minimize(loss)

	return [[x, t, y], model]

def data_construct():
	X=[[0,0],[0,1],[1,0],[1,1]]
	Y=[[0],[1],[1],[0]]
	X=np.array(X).astype('int16')
	Y=np.array(Y).astype('int16')

	return [X, Y]

class flush_display(object):
	def __init__(self):
		return

	def flush(self, res):
		res = np.hstack(res)
		str = '\r\x1b[2K'+np.array2string(res)
		print(str, end='')

	def __enter__(self):
		return self.flush

	def __exit__(self, *exc_info):
		print('')

def main():
	[[x, t, y], model] =    model_building(D_input, D_label, D_hidden)
	[X, Y]             =    data_construct()
	sess               = tf.InteractiveSession()
	                     tf.global_variables_initializer().run()

	#train
	with flush_display() as flush:
		for i in range(T):
			sess.run(model, feed_dict={x:X, t:Y})
			if (i % 200) is 0:
				flush(sess.run(y, feed_dict={x:X}))

if __name__ == '__main__':
    main()