import tensorflow as 
import numpy as np

from TradeAlgorithm import calc_score


class DQN:
	tran_size = 12*24
	BATCH_SIZE = 4
	def __init__(self,session,data_dic,data_len,possess,n_act,BATCH_SIZE):
		self.n_act = n_act
		self.session = session
		self.BATCH_SIZE = BATCH_SIZE
		self.possess = possess
		self.data_dic = data_dic

		self.X = tf.placeholder(tf.float32,[BATCH_SIZE,None])
		self.Y = tf.placeholder(tf.float32,[None])
		self.A = tf.placeholder(tf.float32,[None])

		self.Q = self.build_network('main')
		self.cost, self.train_op = self.build_op()
		self.target_Q = self.build.network('target')

	def build_network(self,name):
		with tf.variable_scope(name):
			model = tf.layers.dense(self.X,1024,activation=tf.nn.relu)
			model = tf.layers.dense(model,512,activation=tf.nn.relu)
			Q = tf.layers.dense(model,self.n_act,activation=None)


		return Q

	def build_op(self):
		one_hot = tf.one_hot(self.X,self.n_act,1.0,0.0)
		Q_value = tf.reduce_sum(tf.multiply(self.Q,one_hot),axis = 1)	
		cost = tf.reduce_mean(tf.square(self.Y - Q_value))
		train_op = tf.train.AdamOptimizer(1e-5).minimize(cost)

		return cost,train_op

	def update_target(self):
		copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def remember(self,cur_index, next_index, possess):
    	self.cur_index = cur_index
    	self.next_index = next_index
    	self.possess = possess

    def get_action(self):
    	Q_value = sess.run(self.Q, feed_dict={X:self.index_to_dic(cur_index)})
    	action_list = []
    	for i in range(BATCH_SIZE):
    		if self.possess[i] > 0:
    			action_list.append(np.argmax(Q_value[i][1:3])+1)
    		else:
    			action_list.append(np.argmax(Q_value[i][:2]))

    	return action_list


    def index_to_dict(self,index_list):
    	tran_dic = []
    	for i in range(BATCH_SIZE):
    		tran_dic.append(data_dic[index_list[i]-tran_size+1]:index_list[i]+1)

    	return tran_dic

    def step(self,action):
    	reward = []
    	for i in range(BATCH_SIZE):
    		reward, possess = TradeAlgorithm.calc_score(data_dic, action, possess)





    	return reward, possess
