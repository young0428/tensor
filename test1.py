import tensorflow as tf
import numpy as np

x_data = np.array([
	[0,0],
	[1,0],
	[1,1],
	[0,0],
	[0,0],
	[0,1]
	])

y_data = np.array([
	[1,0,0],
	[0,1,0],
	[0,0,1],
	[1,0,0],
	[1,0,0],
	[0,0,1]
	])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,10],-1.,1.))
W2 = tf.Variable(tf.random_uniform([10,3],-1.,1.))
"""
b1 = tf.Variables(tf.zeros([10]))
b2 = tf.Variables(tf.zeros([3]))
"""

L1 = tf.matmul(X,W1)
L1 = tf.nn.relu(L1)

model = tf.matmul(L1,W2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model))


optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print('='*10 + 'Learning_start' + '='*10)

for i in range(100):
	
	sess.run(train_op,feed_dict={X:x_data,Y:y_data})

	if (i+1)%10 == 0:
		print(i+1 , sess.run(cost,feed_dict={X:x_data,Y:y_data}))

prediction = tf.argmax(model,axis=1)

print(sess.run(prediction,feed_dict={X:x_data}))
print(sess.run(tf.argmax(Y,axis=1),feed_dict = {Y:y_data}))
print('='*10 + 'learning_done' + '='*10)

sess.close()

 