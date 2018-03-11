import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data",one_hot=True)



total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28*28
n_noise = 128

X = tf.placeholder(tf.float32,[None,n_input])
Z = tf.placeholder(tf.float32,[None,n_noise])

G_w1 = tf.Variable(tf.random_normal([n_noise,n_hidden],stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_w2 = tf.Variable(tf.random_normal([n_hidden,n_input],stddev=0.01))
G_b2 = tf.variable(tf.zeros([n_input]))

D_w1 = tf.Variable(tf.random_normal([n_input,n_hidden],stddev=0.01))
D_w2 = tf.Variable(tf.random_normal([n_input,n_hidden],stddev=0.01))

def generator(noise_z):
	hidden = tf.nn.relu(tf.matmul(noise_z,G_w1)+G_b1)
	output = tf.nn.sigmoid(tf.matmul(hidden,G_w2)+G_b2)
	return output


def discriminator(inputs):
	hidden = tf.nn.relu(tf.matmul(inputs,D_w1))
	output = tf.nn.sigmoid(tf.matmul(hidden,D_w2))
	return output

def get_noise(batch_size,n_noise):
	return np.random.normal(size=(batch_size,n_noise))


G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)



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









