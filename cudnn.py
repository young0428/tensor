import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import time
import mkstockinput
from tensorflow.python.client import device_lib



batch_size = 150
day = 150
n_hidden = 1024
n_class = 13*30

X = tf.placeholder(tf.float32,[None,5,n_class])
Y = tf.placeholder(tf.float32,[None,n_class])
e_step = tf.Variable(0,trainable=False, name='e_step') 

def generator(inputs):
	
	with tf.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
		lstm = tf.contrib.cudnn_rnn.CudnnLSTM(4,n_hidden)
		outputs,states = lstm(inputs)
		outputs = tf.transpose(outputs,[1,0,2])
		outputs = outputs[-1]
		output = tf.layers.dense(outputs,512,activation=tf.nn.relu)
		output = tf.layers.dense(output,1024,activation=tf.nn.relu)
		output = tf.layers.dense(output,n_class,activation=None)
		




	return output
"""
def discriminator(inputs,reuse=None):
	with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:
		
		hidden = tf.layers.dense(inputs,512,activation=tf.nn.leaky_relu)
		hidden = tf.layers.dense(hidden,1024,activation=tf.nn.leaky_relu)
		hidden = tf.layers.dense(hidden,1024,activation=tf.nn.leaky_relu)
		output_y = tf.layers.dense(hidden,1,activation=None)

	return output_y
"""







		






G = generator(X)


loss_G = tf.reduce_mean(tf.square(G-Y))

#vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')

#train_D = tf.train.AdamOptimizer(0.002).minimize(loss_D,var_list=vars_D)
train_G = tf.train.AdamOptimizer(0.002).minimize(loss_G,var_list=vars_G)



sess = tf.Session()
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('../res_save/cu_model')



if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
	print('Model Restored!')
	saver.restore(sess,ckpt.model_checkpoint_path)
	print('e_step : ' ,sess.run(e_step))
else:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)




episode_step = sess.run(e_step)





for episode_step in range(episode_step+1,episode_step+100001):
	f = open('./stockdata/code_list.txt','r')
	code_list = f.readlines()
	cnt = 0
	for code in code_list:
		code = code[:-1]
		code_list[cnt] = code
		cnt+=1
	f.close()

	code_list = random.sample(code_list,len(code_list))
	s_num = 0
	err = 0
	X_input = []
	Y_input = []

	while(s_num<batch_size):
		ck,X_val,Y_val = mkstockinput.mk_data(code_list[err+s_num],day)
		if ck:
			X_input.append(X_val)
			Y_input.append(Y_val)
			s_num += 1
		else:
			err+=1

	
	loss_D_val,loss_G_val = 0,0


	_,loss_G_val = sess.run([train_G,loss_G],feed_dict={X:mkstockinput.x_to_rate(X_input),Y:mkstockinput.y_to_rate(Y_input,X_input)})




	print('Epoch : ', '%-4d   '%(episode_step),'loss : %-9.4f'%(loss_G_val))

	if(episode_step%200 == 0):
		add_op = tf.assign(e_step,episode_step)
		sess.run(add_op)
		saver.save(sess,'../res_save/cu_model/stock.ckpt')

		gene_y = sess.run([G],feed_dict={X:mkstockinput.x_to_rate(X_input)})
		gene_y = gene_y[-1]
		gene_y = mkstockinput.rate_to_x(gene_y,X_input)
		real_y = Y_input
		axis_x = range(len(gene_y[0]))

		plt.figure(figsize=(16,12))

		for k in range(9):
			plt.subplot(3,3,k+1)
			plt.plot(axis_x,gene_y[k])
			plt.plot(axis_x,real_y[k])
			plt.legend(['Gene','Real'])


		plt.savefig('../res_save/cu_result/{}.png'.format(str(episode_step).zfill(5)))

		





		












	






