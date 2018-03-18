import tensorflow as tf


class DQN:
	tran_size = 12*12
	BATCH_SIZE = 4
	def __init__(self,session,transaction,n_act):
		self.n_act = n_act
		self.session = session
		self.tran = []

		self.X = tf.placeholder(tf.float32,[None])
		self.Y = tf.placeholder(tf.float32,[None])
		self.A = tf.placeholder(tf.float32,[None])

		self.Q = self.build_network('main')
		self.cost, self.train_op = self.build_op()
		self.target_Q = self.build.network('target')

	def build_network(self,name):
		with tf.variable_scope(name):
			model = tf.layers.dense(self.X,256,activation=tf.nn.relu)
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

    def get_action(self):
    	Q_value = session.run(self.Q,feed_dict={X:self.tran})
    	action = np.argmax()