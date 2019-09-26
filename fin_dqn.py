import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import time
import dqn_input
from tensorflow.python.client import device_lib

UPDATE_CNT = 0 
UPDATE_TERM = 40
batch_size = 50
day = 30
n_hidden = 256
n_class = 30
n_action = 2
learning_rate = 1e-9
dropout = 0.02
real_score = [0,]*batch_size



X_price = tf.placeholder(tf.float32,[None,13*day])
X_volume = tf.placeholder(tf.float32,[None,13*day])
Y = tf.placeholder(tf.float32,[None])
A = tf.placeholder(tf.int64,[None])
e_step = tf.Variable(0,trainable=False, name='e_step')


los_summ = tf.placeholder(tf.float32,shape=())
tf.summary.scalar('reward',tf.div(los_summ,10))

def build_net(name):
	
	with tf.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):

		X = tf.reshape(X_price,[batch_size,day/2,13*2])
		lstm1 = tf.contrib.cudnn_rnn.CudnnLSTM(3,n_hidden,dropout=dropout)
		outputs1,states = lstm(X)
		outputs1 = tf.transpose(outputs1,[1,0,2])
		outputs1 = outputs1[-1]
		L1 = tf.layers.dense(outputs1,32,activation=None)

		X = tf.reshape(X_volume,[batch_size,day/2,13*2])
		lstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(3,n_hidden,dropout=dropout)
		outputs2,states = lstm(inputs)
		outputs2 = tf.transpose(outputs,[1,0,2])
		outputs2 = outputs2[-1]

		L2 = tf.layers.dense(outputs2,128,activation=tf.nn.relu)


		L = tf.concat([L1,L2],1)
		
		L = tf.layers.dense(L,1024,activation=tf.nn.relu)
		output = tf.layers.dense(L,n_action,activation=None)

		

	return output


def get_action(price,volume):
	Q_value = sess.run(Q,feed_dict={X_price:dqn_input.x_to_rate(price),X_volume:volume})
	action_list = np.argmax(Q_value,axis = 1)

	return action_list


def game_play(act,x_price,e_index,poss_index,poss_bool,stp,batch_size):
	price_rate = dqn_input.x_to_rate(x_price)
	score = [0,]*batch_size
	for i in range(batch_size):
		if act[i] == 0:
			score[i] = price_rate[i][e_index+stp] - price_rate[i][e_index]
			if not poss_bool[i] :
				poss_bool[i] = True
				poss_index[i] = e_index


		elif act[i] == 1:
			score[i] = 0
			if poss_bool[i] : 
				real_score[i] += x_price[i][poss_index[i]] - x_price[i][e_index]
				poss_bool[i] = False
				poss_index[i] = -1

	return score




		






Q = build_net('main')
target_Q = build_net('target')

one_hot = tf.one_hot(A,n_action,1.0,0.0)
Q_value = tf.reduce_sum(tf.multiply(Q,one_hot),axis=1)

loss = tf.reduce_mean(tf.losses.absolute_difference(labels=Y,predictions=Q_value))


#vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')
vars_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='main')

#train_D = tf.train.AdamOptimizer(0.002).minimize(loss_D,var_list=vars_D)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=vars_main)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('../res_save/cuv2_model')

def update_target():
	copy_op = []

	main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
	target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

	for main_var, target_var in zip(main_vars, target_vars):
		copy_op.append(target_var.assign(main_var.value()))

	sess.run(copy_op)



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
	X_data = []
	X_vol_data = []

	while(s_num<batch_size):
		ck,X_val,X_volume_val = dqn_input.mk_data(code_list[err+s_num],day)
		if ck:
			X_data.append(X_val)
			X_vol_data.append(X_volume_val)
			s_num += 1
		else:
			err+=1

	step = 3 # 2시간 간격 업데이트
	possess_index = [-1,]*batch_size
	possess_bool = [False,]*batch_size
	start_index = 0
	end_index = 13*30
	action = [0,]*batch_size
	score_sum = [0,]*batch_size
	loss_sum = 0

	for game_start in range(int(13*30/3)):
		X_PRICE_INPUT = np.array(X_data)[:,start_index:end_index]
		X_VOLUME_INPUT = np.array(X_vol_data)[:,start_index:end_index]
		input_Y = [0,]*batch_size
		#  0:구매, 1:판매
		if UPDATE_CNT < 50 and episode_step == 0:
			for j in range(batch_size):
				action[j] = random.randrange(0,2)
		else:
			action = get_action(X_PRICE_INPUT,X_VOLUME_INPUT)	

		reward = game_play(action,X_data,end_index,possess_index,possess_bool,step,batch_size)

		target_Q_value = sess.run(target_Q,feed_dict={X_price:dqn_input.x_to_rate(np.array(X_data)[:,start_index+step:end_index+step]),
													  X_volume:np.array(X_vol_data)[:,start_index+step:end_index+step]})
		for i in range(batch_size):
			input_Y[i] = np.max(target_Q_value[i]) + reward[i]


		avg_cost,_ = sess.run([loss,train_op],feed_dict={X_price:dqn_input.x_to_rate(X_PRICE_INPUT),
														 X_volume:X_VOLUME_INPUT,
														 Y:input_Y,
														 A:action})

		for i in range(batch_size):
			score_sum[i] = score_sum[i] + reward[i]

		UPDATE_CNT+=1
		if UPDATE_CNT%UPDATE_TERM == 0:
			update_target()

		start_index += step
		end_index += step
		loss_sum += avg_cost
	print("Epoch : %-5d"%(episode_step))
	bene_cnt = 0

	for i in range(batch_size):
		if possess_bool:
			real_score[i] += X_data[i][possess_index[i]] - X_data[i][end_index]

	for i in range(batch_size):
		
		if real_score[i] > 0:
			print("%2d : %-5.2f < ---- benefit "%(i,real_score[i]))
			bene_cnt+=1
		else:
			print("%2d : %-5.2f                "%(i,real_score[i]))

	print("benefit ratio : %2.3f   loss : %-9.4f"%(bene_cnt/batch_size*100,float(loss_sum/(13*30/3))))
	q_l = sess.run(Q,feed_dict={X_price:dqn_input.x_to_rate(X_PRICE_INPUT), X_volume:X_VOLUME_INPUT})
	print(q_l[0])

	real_score = [0,]*batch_size


	

	if (episode_step+1) % 10 == 0:
		summary = sess.run(summary_merged, feed_dict = {los_summ:float(loss_sum/13*30/3)})
		writer.add_summary(summary,episode_step)
		loss_val = 0



	if(episode_step%10 == 0):
		add_op = tf.assign(e_step,episode_step)
		sess.run(add_op)
		saver.save(sess,'../res_save/cuv2_model/stock.ckpt')

	

		










	





