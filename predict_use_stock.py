import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import time
from mkstockinput import mk_data



batch_size = 100
day = 30
n_hidden = 512
n_class = 13*10

X = tf.placeholder(tf.float32,[None,day,13])
Y = tf.placeholder(tf.float32,[None,n_class])
e_step = tf.Variable(0,trainable=False, name='e_step')

def generator(inputs):
	
	with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
		cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
		outputs,states = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)
		outputs = tf.transpose(outputs,[1,0,2])
		outputs = outputs[-1]
		output = tf.layers.dense(outputs,256,activation=tf.nn.leaky_relu)
		output = tf.layers.dense(output,512,activation=tf.nn.leaky_relu)
		output = tf.layers.dense(output,128,activation=tf.nn.leaky_relu)
		output = tf.layers.dense(output,n_class,activation=None)

	return output

def discriminator(inputs,reuse=None):
	with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE) as scope:
		
		hidden = tf.layers.dense(inputs,512,activation=tf.nn.leaky_relu)
		hidden = tf.layers.dense(hidden,1024,activation=tf.nn.leaky_relu)
		hidden = tf.layers.dense(hidden,1024,activation=tf.nn.leaky_relu)
		output_y = tf.layers.dense(hidden,1,activation=None)

	return output_y


		






G = generator(X)
D_real = discriminator(Y)
D_gene = discriminator(G,True)

loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real,labels=tf.ones_like(D_real)))
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene,labels=tf.zeros_like(D_gene)))

loss_D = loss_D_gene + loss_D_real

loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene,labels=tf.ones_like(D_gene)))

vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')

train_D = tf.train.AdamOptimizer(0.002).minimize(loss_D,var_list=vars_D)
train_G = tf.train.AdamOptimizer(0.002).minimize(loss_G,var_list=vars_G)

sess = tf.Session()
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./model')

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
		ck,X_val,Y_val = mk_data(code_list[err+s_num])
		if ck:
			X_input.append(X_val)
			Y_input.append(Y_val)
			s_num += 1
		else:
			err+=1

	
	loss_D_val,loss_G_val = 0,0

	_,loss_D_val = sess.run([train_D,loss_D],feed_dict={X:X_input,Y:Y_input})
	_,loss_G_val = sess.run([train_G,loss_G],feed_dict={X:X_input})




	print('Epoch : ', '%-4d'%(episode_step),'D loss : %-9.4f || G loss : %-9.4f'%(loss_D_val,loss_G_val))

	if(episode_step%200 == 0):
		add_op = tf.assign(e_step,episode_step)
		sess.run(add_op)
		saver.save(sess,'./model/stock.ckpt')

		gene_y = sess.run([G],feed_dict={X:X_input})
		gene_y = gene_y[-1]
		real_y = Y_input
		axis_x = range(len(gene_y[0]))

		plt.figure(figsize=(12,9))

		plt.plot(axis_x,gene_y[0])
		plt.plot(axis_x,real_y[0])
		plt.legend(['Gene','Real'])

		plt.savefig('./result2/{}.png'.format(str(episode_step).zfill(4)))

		





		












	






