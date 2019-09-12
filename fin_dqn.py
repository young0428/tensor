import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import time
import dqn_input
from tensorflow.python.client import device_lib



batch_size = 50
day = 30
n_hidden = 512
n_class = 30
n_action = 2
learning_rate = 0.0000001
dropout = 0.005


X_price = tf.placeholder(tf.float32,[None,13])
X_volume = tf.placeholder(tf.int64,[None,1])
Y = tf.placeholder(tf.float32,[None,n_class])
A = tf.placeholder(tf.int64,[None])
e_step = tf.Variable(0,trainable=False, name='e_step')


los_summ = tf.placeholder(tf.float32,shape=())
tf.summary.scalar('reward',tf.div(los_summ,10))

def build_net(name):
	
	with tf.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):

		L1 = tf.layers.conv1d(X_price,16,6,actiavtion=tf.nn.relu)
		L1 = tf.nn.dropout(L,dropout)
		L1 = tf.layres.conv1d(L1,32,3,activation=tf.nn.relu)
		L1 = tf.nn.dropout(L,dropout)
		L1 = tf.layers.conv1d(L1,64,3,actiavtion=tf.nn.relu)
		L1 = tf.layers.dense(L1,512,activation=None)

		L2 = tf.layers.dense(X_volume,128,activation=None)

		L = tf.concat([L1,L2],1)
		L = tf.layers.dense(L,512,activation=tf.nn.relu)
		L = tf.layers.dense(L,1024,activation=tf.nn.relu)
		L = tf.layers.dense(L,1024,activation=tf.nn.relu)
		output = tf.layers.dense(L,n_action,activation=None)

		

	return output






		






Q = build_net('main')
target_Q = build_net('target')

one_hot = tf.one_hot(A,n_action,1.0,0.0)
Q_value = tf.reduce_sum(tf.multiply(Q,one_hot),axis=1)

loss = tf.reduce_mean(tf.losses.absoulute_difference(labels=Y,predictions=Q_value))


#vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')
vars_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='main')

#train_D = tf.train.AdamOptimizer(0.002).minimize(loss_D,var_list=vars_D)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=vars_main)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('../res_save/cuv2_model')



if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
	print('Model Restored!')
	saver.restore(sess,ckpt.model_checkpoint_path)
	print('e_step : ' ,sess.run(e_step))
else:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)




episode_step = sess.run(e_step)

writer = tf.summary.FileWriter('./logs',sess.graph)
summary_merged = tf.summary.merge_all()



loss_val = 0

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
		ck,X_val,Y_val = dqn_input.mk_data(code_list[err+s_num],day)
		if ck:
			X_input.append(X_val)
			Y_input.append(Y_val)
			s_num += 1
		else:
			err+=1

	
	loss_D_val,loss_G_val = 0,0



	_,loss_G_val = sess.run([train_G,loss_G],feed_dict={X:mkstockinput.x_to_rate(X_input),Y:mkstockinput.y_to_rate(Y_input,X_input)})




	print('Epoch : ', '%-4d   '%(episode_step),'loss : %-9.4f'%(loss_G_val))

	loss_val = loss_val + loss_G_val

	if (episode_step+1) % 10 == 0:
		summary = sess.run(summary_merged, feed_dict = {los_summ:loss_val})
		writer.add_summary(summary,episode_step)
		loss_val = 0



	if(episode_step%50 == 0):
		add_op = tf.assign(e_step,episode_step)
		sess.run(add_op)
		saver.save(sess,'../res_save/cuv2_model/stock.ckpt')

	if(episode_step%200 == 0):
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


		plt.savefig('../res_save/cuv2_result/{}.png'.format(str(episode_step).zfill(5)))

		





		







	





