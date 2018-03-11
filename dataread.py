import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack = True, dtype='float32')

#기본데이터 입력
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])


#학습횟수를 저장하기위한 변수선언
global_step = tf.Variable(0, trainable=False, name='global_step')


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#신경망 구성 은닉계층

with tf.name_scope('layer1'):
	w1 = tf.Variable(tf.random_uniform([2,10],-1.,1.),name='w1')
	L1 = tf.nn.relu(tf.matmul(X,w1))
with tf.name_scope('layer2'):
	w2 = tf.Variable(tf.random_uniform([10,20],-1.,1.),name='w2')
	L2 = tf.nn.relu(tf.matmul(L1,w2))
with tf.name_scope('ouput'):
	w3 = tf.Variable(tf.random_uniform([20,3],-1.,1.),name ='w3')
	model = tf.matmul(L2,w3)
with tf.name_scope('optimizer'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
	train_op = optimizer.minimize(cost, global_step = global_step)



tf.summary.scalar('cost',cost)
tf.summary.histogram('weight',w1)


sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())




#체크포인트 파일 체크

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
	saver.restore(sess, ckpt.model_checkpoint_path)
else:
	sess.run(tf.global_variables_initializer())


writer = tf.summary.FileWriter('./logs',sess.graph)

for i in range(50):
	sess.run(train_op,feed_dict={X:x_data,Y:y_data})

	print('Step: %d ' % sess.run(global_step),
		  'Cost: %.5f'% sess.run(cost,feed_dict = {X:x_data,Y:y_data}))

	summary = sess.run(tf.summary.merge_all(),feed_dict={X:x_data,Y:y_data})
	writer.add_summary(summary,global_step=sess.run(global_step))



saver.save(sess, './model/dnn.ckpt', global_step = global_step)

prediction = tf.argmax(model,1)
actu = tf.argmax(Y,1)

print('prediction:', sess.run(prediction,feed_dict={X:x_data}))
print('Aactually :', sess.run(actu,feed_dict={Y:y_data}))

print("done")

