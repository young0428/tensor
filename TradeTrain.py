import tensorflow as tf
import numpy as np
import datetime
import Trademodule
import random

from trademodel import DQN
from TradeAlgorithm import calc_score


def train():
	time_step = 0
	episode_step = 0
	n_act = 2
	INPUT_SIZE = 12*24
	period = 12*24*30
	epoch_SIZE = 10000
	BATCH_SIZE = 8
	UPDATE_TERM = 12
	TARGET_UPDATE_INTERVAL = 120
	action = []
	possess = []
	total_reward = []
	total_reward_list = []
	total_real_reward = []
	trade_check = []
	trade_cnt = []


	
	e_step = tf.Variable(0, trainable=False, name = 'e_step')
	t_step = tf.Variable(0,trainable=False, name='t_step')
	
	sess = tf.Session()
	next_transaction_index = []
	

	# 결과 텐서보드


	data_len, data_dic = Trademodule.get_all_data_timestamp()
	for i in range(data_len):
		data_dic[i][1] = int(float(data_dic[i][1])*1000)
		data_dic[i][0] = int(data_dic[i][0])
	
	brain = DQN(sess,data_dic,data_len,possess,n_act,BATCH_SIZE,UPDATE_TERM)


	rewards = tf.placeholder(tf.float32,[None])
	tf.summary.scalar('avg.reward/ep.',tf.reduce_mean(rewards))



	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state('./model')
	
	init_op = tf.global_variables_initializer()
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Model restored!!")
		saver.restore(sess,ckpt.model_checkpoint_path)
		print('Stored episode_step :',sess.run(e_step))
	else :
		sess.run(init_op)


	writer = tf.summary.FileWriter('./logs',sess.graph)
	summary_merged = tf.summary.merge_all()

	episode_step = sess.run(e_step)
	time_step = sess.run(t_step)




	for _ in range(BATCH_SIZE):
		next_transaction_index.append(0)
		possess.append(0) 
		action.append(0)
		total_reward.append(0)
		total_real_reward.append(0)
		trade_cnt.append(0)
		trade_check.append(0)


	
	for i in range(epoch_SIZE):
		cur_transaction_index = Trademodule.get_startpoint(data_dic,INPUT_SIZE,period,BATCH_SIZE)
		avg = 0
		for t in range(BATCH_SIZE):
			total_reward[t] = 0
			total_real_reward[t] = 0
			possess[t] = 0
			trade_cnt[t] = 0
			trade_check[t] = 0

		for j in range(int(period/UPDATE_TERM)): #트레이닝 진행
			for index in range(BATCH_SIZE): #장부 6개 업데이트
				next_transaction_index[index] = cur_transaction_index[index]+UPDATE_TERM

			brain.remember(cur_transaction_index, next_transaction_index,possess)

			# 0:구매 1:판매
			if episode_step == 0 and time_step < 100000 :
				for t in range(BATCH_SIZE):
					action[t] = random.randrange(0,10)
					if action[t] > 1:
						action[t] = 0

			else:
				action = brain.get_action()

			if time_step % TARGET_UPDATE_INTERVAL == 0:
				brain.update_target()

			for t in range(BATCH_SIZE):
				if possess[t] > 0 :
					trade_check[t] = 1

			for t in range(BATCH_SIZE):
				if possess[t] > 0 :
					if (action[t] == 0 and j+1==int(period/UPDATE_TERM)) or (data_dic[cur_transaction_index[t]][1]/data_dic[possess[t]][1] < 0.8) :
						action[t] = 1

			real_reward,reward, possess = brain.step(action)

			for t in range(BATCH_SIZE):
				if possess[t] == 0 and trade_check[t] == 1:
					trade_check[t] = 0
					trade_cnt[t] += 1

			for t in range(BATCH_SIZE):
				total_reward[t] = total_reward[t] + reward[t]
				total_real_reward[t] = total_real_reward[t] + real_reward[t]

				avg += reward[t]
			brain.train(reward,action)
			cur_transaction_index = next_transaction_index
			time_step += 1 





		total_reward_list.append(avg/BATCH_SIZE)
		episode_step = episode_step + 1
			

		print('학습횟수 : %d' % episode_step)
		if (i+1) % 2 == 0:
			summary = sess.run(summary_merged, feed_dict = {rewards:total_reward_list})
			writer.add_summary(summary,time_step)
			total_reward_list = []
		if (i+1) % 20 == 0:
			add_op_1 = tf.assign(e_step,episode_step)
			add_op_2 = tf.assign(t_step,time_step)
			sess.run([add_op_1,add_op_2])
			saver.save(sess, './model/trade.ckpt',global_step=episode_step)


		
		for j in range(BATCH_SIZE):
			if total_real_reward[j] > 0 :
				print('%02d : %13f    %13f      %02d   <-- benefit!'%(j+1,total_reward[j],total_real_reward[j],trade_cnt[j]))
			else:
				print('%02d : %13f    %13f      %02d'%(j+1,total_reward[j],total_real_reward[j],trade_cnt[j]))
		print('평균 점수 : %06f' % (avg/BATCH_SIZE))
		print('\n\n')
			

train()








