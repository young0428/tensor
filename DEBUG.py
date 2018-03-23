import tensorflow as tf

a = tf.Variable(1,trainable = False, name = 'a')
b = tf.Variable(2,trainable = False, name = 'b')
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
	a = tf.assign(a,a+b)
	sess.run(a)
	saver.save(sess,'./test_debug/test.ckpt',global_step = i)

sess.close()	
