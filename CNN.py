import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data",one_hot=True)


X = tf.placeholder(tf.float32,[None,28,28,1])
Y = tf.placeholder(tf.float32,[None,10])
d_prob = tf.placeholder(tf.float32)

L1 = tf.layers.conv2d(X,32,[3,3])
L1 = tf.layers.max_pooling2d(L1,[2,2],[2,2])
L1 = tf.nn.dropout(L1,d_prob)

L2 = tf.layers.conv2d(L1,64,[3,3])
L2 = tf.layers.max_pooling2d(L2,[2,2],[2,2])
L2 = tf.nn.dropout(L2,d_prob)

L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3,256,activation=tf.nn.relu)
L3 = tf.nn.dropout(L3,d_prob)


model = tf.layers.dense(L3,10,activation=None)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=model))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(10):
	total_cost = 0

	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = batch_xs.reshape(-1,28,28,1)
		_, cost_val = sess.run([optimizer,cost],feed_dict = {X:batch_xs,Y:batch_ys,d_prob:0.7})
		total_cost += cost_val


	print('Epoch:', '%04d'%(epoch+1),
		  'Avg.cost = ','{:.3f}'.format(total_cost/total_batch))


print('Train_done!!')

is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

print('Accu:',sess.run(accuracy,feed_dict={X:mnist.test.images.reshape(-1,28,28,1),Y:mnist.test.labels,d_prob:1}))









