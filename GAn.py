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
G_b2 = tf.Variable(tf.zeros([n_input]))

D_w1 = tf.Variable(tf.random_normal([n_input,n_hidden],stddev=0.01))
D_w2 = tf.Variable(tf.random_normal([n_hidden,1],stddev=0.01))

def generator(noise_z):
	hidden = tf.nn.relu(tf.matmul(noise_z,G_w1)+G_b1)
	output = tf.sigmoid(tf.matmul(hidden,G_w2)+G_b2)
	return output


def discriminator(inputs):
	hidden = tf.nn.relu(tf.matmul(inputs,D_w1))
	output = tf.matmul(hidden,D_w2)
	return output

def get_noise(batch_size,n_noise):	
	return np.random.normal(size=(batch_size,n_noise))


G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gene,labels=tf.zeros_like(D_gene)))
loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real,labels=tf.ones_like(D_real)))

loss_D = loss_D_gene + loss_D_real

loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gene,labels=tf.ones_like(D_gene)))

var_list_D = [D_w1,D_w2]
var_list_G = [G_w1,G_w2,G_b1,G_b2]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D,var_list = var_list_D)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G,var_list = var_list_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
val_loss_D, val_loss_G = 0,0







for epoch in range(total_epoch):
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		noise = get_noise(batch_size,n_noise)
		_,val_loss_D = sess.run([train_D,loss_D],feed_dict={X:batch_xs,Z:noise})
		_,val_loss_G = sess.run([train_G,loss_G],feed_dict={Z:noise})


	print('Epoch:', '%04d'%(epoch+1),
		  'D_loss = ','{:.3f}'.format(val_loss_D),
		  'G_loss = ','{:.3f}'.format(val_loss_G))
	if (epoch%10) == 0 :
		sample_size = 10
		noise = get_noise(sample_size,n_noise)
		samples = sess.run(G,feed_dict={Z:noise})

		fig, ax = plt.subplots(1,sample_size,figsize=(sample_size,1))
		for i in range(sample_size):
			ax[i].set_axis_off()
			ax[i].imshow(np.reshape(samples[i],(28,28)))

		plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)),bbox_inches='tight')

		plt.close(fig)


print('Train_done!!')





