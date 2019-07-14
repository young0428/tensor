import tensorflow as tf
import numpy as np

import TradeAlgorithm


class DQN:
	def __init__(self,session,data_dic,data_len,possess,n_act,BATCH_SIZE,UPDATE_TERM):
		self.n_act = n_act
		self.sess = session
		self.BATCH_SIZE = BATCH_SIZE
		self.possess = possess
		self.tran_size = 12*24
		self.data_dic = data_dic
		self.action = []
		self.UPDATE_TERM = UPDATE_TERM

		self.input_X = tf.placeholder(tf.float32,[None,self.tran_size,1])
		self.input_A = tf.placeholder(tf.int64,[None])
		self.input_Y = tf.placeholder(tf.float32,[None])
		self.keep_prob = tf.placeholder(tf.float32)

		self.Q = self.build_network('main')
		self.cost, self.train_op = self.build_op()
		self.target_Q = self.build_network('target')

	def build_network(self,name):
		with tf.variable_scope(name):
			model = tf.layers.conv1d(self.input_X,64,6,activation=tf.nn.relu)
			model = tf.nn.dropout(model, self.keep_prob)
			model = tf.layers.conv1d(model,128,3,activation=tf.nn.relu)
			model = tf.nn.dropout(model,self.keep_prob)
			model = tf.layers.dense(model,512,activation=tf.nn.relu)

			Q = tf.layers.dense(model,self.n_act,activation=None)


		return Q

	def build_op(self):
		one_hot = tf.one_hot(self.input_A,self.n_act,1.0,0.0)
		Q_value = tf.reduce_sum(tf.multiply(self.Q,one_hot),axis = 1)	
		cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
		train_op = tf.train.AdamOptimizer(0.00002).minimize(cost)

		return cost,train_op

	def update_target(self):
		copy_op = []

		main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
		target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

		for main_var, target_var in zip(main_vars, target_vars):
			copy_op.append(target_var.assign(main_var.value()))

		self.sess.run(copy_op)

	def remember(self,cur_index, next_index, possess):
		self.cur_index = cur_index
		self.next_index = next_index
		self.possess = possess

	def get_action(self):
		Q_value = self.sess.run(self.Q, feed_dict={self.input_X:self.index_to_dic(self.cur_index),self.keep_prob:0.9})
		action_list = np.argmax(Q_value,axis = 1)

		return action_list


	def index_to_dic(self,index_list):
		col = []
		a = []
		ud = 0
		for i in range(self.BATCH_SIZE):
			a = []
			tran_dic = self.data_dic[(index_list[i]-self.tran_size):index_list[i]+1]	
			for j in range(len(tran_dic)-1):
				md = []
				md.append(tran_dic[j+1][1] - tran_dic[j][1])
				a.append(md)

			col.append(a)


		return col

	def step(self,action):
		real_reward, reward, possess = TradeAlgorithm.calc_score(self.data_dic, self.cur_index, action, self.possess, self.BATCH_SIZE, self.UPDATE_TERM)
		return real_reward,reward, possess

	def train(self,reward,action):
		target_Q_value = self.sess.run(self.target_Q,feed_dict={self.input_X:self.index_to_dic(self.next_index),self.keep_prob:0.9})
		Y = [] 

		for i in range(self.BATCH_SIZE):
			Y.append(reward[i] + np.max(target_Q_value[i]))

		self.sess.run(self.train_op,feed_dict={self.input_X:self.index_to_dic(self.cur_index),
											   self.input_Y:Y,
											   self.input_A:action,
											   self.keep_prob:0.9})

