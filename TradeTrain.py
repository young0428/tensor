import tensorflow as tf
import numpy as np
import datetime
import Trademodule
import random

from trademodel import DQN


def train():
	time_step = 0
	episode_step = 0
	n_act = 2
	INPUT_SIZE = 12*24
	period = 12*24*10
	epoch_SIZE = 10000
	BATCH_SIZE = 4
	UPDATE_TERM = 6
	TARGET_UPDATE_INTERVAL = 100
	action = []
	possess = []
	total_reward = []
	total_reward_list = []

	
	
	e_step = tf.Variable(0, trainable=False, name = 'e_step')
	t_step = tf.Variable(0,trainable=False, name='t_step')
	
	sess = tf.Session()
	next_transaction_index = []
	

	# 결과 텐서보드


	data_len, data_dic = Trademodule.get_all_data_timestamp()
	for i in range(data_len):
		data_dic[i][1] = float(data_dic[i][1])
	
	brain = DQN(sess,data_dic,data_len,possess,n_act,BATCH_SIZE)


	rewards = tf.placeholder(tf.float32,[None])
	tf.summary.scalar('avg.reward/ep.',tf.reduce_mean(rewards))



	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state('./model')
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Model restored!!")
		saver.restore(sess,ckpt.model_checkpoint_path)
		print('Stored episode_step :',sess.run(e_step))
	else :
		sess.run(tf.global_variables_initializer())


	writer = tf.summary.FileWriter('./logs',sess.graph)
	summary_merged = tf.summary.merge_all()

	episode_step = sess.run(e_step)
	time_step = sess.run(t_step)




	for _ in range(BATCH_SIZE):
		next_transaction_index.append(0)
		possess.append(0) 
		action.append(0)
		total_reward.append(0)


	
	for i in range(epoch_SIZE):
		cur_transaction_index = Trademodule.get_startpoint(data_dic,INPUT_SIZE,period,BATCH_SIZE)
		avg = 0
		for t in range(BATCH_SIZE):
			total_reward[t] = 0

		for j in range(int(period/UPDATE_TERM)): #트레이닝 진행
			for index in range(BATCH_SIZE): #장부 6개 업데이트
				next_transaction_index[index] = cur_transaction_index[index]+UPDATE_TERM

			brain.remember(cur_transaction_index, next_transaction_index,possess)

			# 0:구매 1:판매
			if e_step == 0 and time_step < 100 :
				for t in range(BATCH_SIZE):
					action[t] = random.randrange(0,2) 
			else:
				action = brain.get_action()

			if time_step % TARGET_UPDATE_INTERVAL == 0:
				brain.update_target()

			reward, possess = brain.step(action)

			for t in range(BATCH_SIZE):
				total_reward[t] = total_reward[t] + reward[t]
				avg += reward[t]
			brain.train(reward,action)
			cur_transaction_index = next_transaction_index
			time_step += 1 

		total_reward_list.append(avg/BATCH_SIZE)
		
		episode_step = episode_step + 1
			

		print('학습횟수 : %d' % episode_step)
		if (i+1) % 10 == 0:
			summary = sess.run(summary_merged, feed_dict = {rewards:total_reward_list})
			writer.add_summary(summary,time_step)
			print(total_reward_list)
			total_reward_list = []
		if (i+1) % 50 == 0:
			add_op_1 = tf.assign(e_step,episode_step)
			add_op_2 = tf.assign(t_step,time_step)
			sess.run([add_op_1,add_op_2])
			saver.save(sess, './model/trade.ckpt',global_step=episode_step)


		
		for j in range(BATCH_SIZE):
			print('%d : %d'%(j,total_reward[j]))
		print('\n\n')
			

train()








