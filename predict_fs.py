import tensorflow as tf
import numpy as np
import random
from datetime import datetime
import time
from mkinput import ck_data



batch_size = 100


X = tf.placeholder(tf.float64,[None,44])
Y = tf.placeholder(tf.float32,[None,4])

e_step = tf.Variable(0,trainable=False, name='e_step')


L = tf.layers.dense(X,2048,activation=tf.nn.leaky_relu)

output = tf.layers.dense(L,4,activation = None)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=output))

optimizer = tf.train.AdamOptimizer(0.00001).minimize(cost)

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
	change_list = []
	x_list = []
	y_list = []
	depth = 4
	err = 0
	s_num = 0
	up = 0
	down = 0
	while(s_num<batch_size):
		ck,change_rate,input_data = ck_data(code_list[err+s_num])
		

		if ck:
			if change_rate > 0:
				up +=1
			else:
				down +=1

			if (up > batch_size/2 and change_rate > 0) or (down > batch_size/2 and change_rate <= 0):
				err += 1
				continue

			change_list.append(change_rate)
			x_list.append(input_data)
			s_num += 1
			if(change_rate > 15):
				y = 3
			elif(change_rate > 0):
				y = 2
			elif change_rate > -10:
				y = 1
			else:
				y = 0
			y_list.append(y)
		else:
			err+=1


	print(up,down)
	targets = np.array(y_list).reshape(-1)
	one_hot_targets = np.eye(depth,dtype=float)[targets]
	y_list = one_hot_targets


	_,cost_val = sess.run([optimizer,cost],feed_dict={X:x_list,Y:y_list})
	print('Epoch : ', '%4d'%(episode_step),'Avg.cost = ','%.3f'%(cost_val/batch_size))
	if(episode_step%50 == 0):
		add_op = tf.assign(e_step,episode_step)
		sess.run(add_op)
		saver.save(sess,'./model/fs.ckpt')
	if(episode_step%50 == 0):
		out = tf.argmax(output,1)
		result = sess.run([out],feed_dict={X:x_list})
		correct = 0
		for i in range(batch_size):
			if(change_list[i] > 12):
				y = 3
			elif(change_list[i] > 0):
				y = 2
			elif change_list[i] > -10:
				y = 1
			else:
				y = 0
			if y == result[0][i]:
				correct += 1
				s = '<- correct!!!'
			else:
				s = ''
			print('%3d'%(change_list[i])+ '   ' + '%2d'%(result[0][i]-2),s)

		print('Accuracy : {:.1f}%'.format(correct/batch_size*100))

		












	






