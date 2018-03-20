import tensorflow as tf
import numpy as np
import datetime
import Trademodule
import random

from trademodel import DQN


def train():
	n_act = 4
	INPUT_SIZE = 12*24
	period = 12*24*10
	epoch_SIZE = 10000
	action = []
	possess = []
	BATCH_SIZE = 4
	UPDATE_TERM = 6
	next_transaction_index = []

	data_len, data_dic = Trademodule.get_all_data_timestamp()
	sess = tf.Session()
	brain = DQN(sess,data_dic,data_len,possess,n_act,BATCH_SIZE)

	for _ in ragne(BATCH_SIZE):
		next_transaction_index.append(0)
		possess.append(False) 
		action.append(0)


	for i in range(epoch_SIZE):
		cur_transaction_index = Trademodule.get_startpoint(data_dic,INPUT_SIZE,period,BATCH_SIZE)
		total_reward = []

		for j in range(period/UPDATE_TERM): #트레이닝 진행
			for index in range(BATCH_SIZE): #장부 6개 업데이트
				next_transaction_index[index] = cur_transaction_index[index]+UPDATE_TERM

			brain.remember(cur_transaction_index, next_transaction_index, possess)

			# 0:구매 1:대기 2:판매 
			if i == 0 and j < 100 :
				for t in range(BATCH_SIZE):
					if not possess[t]:
						action[t] = random.randrange(0,2)
					else
						action[t] = random.randrange(1,3)

			else
				action = brain.get_action()








