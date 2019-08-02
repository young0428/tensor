import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data",one_hot=True)


X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])
prob = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([784,256],stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X,w1))
L1 = tf.nn.dropout(L1,prob)


L2 = tf.layers.dense(L1,256,activation=tf.nn.relu)
L2 = tf.nn.dropout(L2,prob)



"""
w2 = tf.Variable(tf.random_normal([512,512],stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1,w2))
L2 = tf.nn.dropout(L2,prob)
"""

model = tf.layers.dense(L2,10,activation=None)

print(model)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=model))

optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

print()


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

batch_size = 1000
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(10):
	total_cost = 0
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		_, cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys,prob:0.8})
		total_cost += cost_val

	print('Epoch:', '%04d'%(epoch + 1),
		   'Avg_cost =','{:.3f}'.format(total_cost/total_batch))

print('Training Done!')


is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:',sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels,prob:1}))


fig = plt.figure()
cnt = 0
false_cnt = 0
print(mnist.test.images)
labels = sess.run(model,feed_dict={X:mnist.test.images,
								   Y:mnist.test.labels,
								   prob: 1})
print(labels[0])
for bl in sess.run(tf.cast(is_correct,tf.float32),feed_dict={X:mnist.test.images,Y:mnist.test.labels,prob:1}):
	if bl == 0:
		subplot = fig.add_subplot(4,5,false_cnt+1)
		subplot.set_xticks([])
		subplot.set_yticks([])
		subplot.set_title('%d' % np.argmax(labels[cnt]))
		subplot.imshow(mnist.test.images[cnt].reshape((28,28)),cmap=plt.cm.gray_r)
		false_cnt+=1

	cnt+=1
	if false_cnt == 20:
		break

plt.show()

"""


fig = plt.figure()

for i in range(10):
	subplot = fig.add_subplot(2,5,i+1)
	subplot.set_xticks([])
	subplot.set_yticks([])
	subplot.set_title('%d' % np.argmax(labels[i]))
	subplot.imshow(mnist.test.images[i].reshape((28,28)),cmap=plt.cm.gray_r)

plt.show()
"""












