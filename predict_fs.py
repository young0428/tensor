import tensorflow as tf
import numpy as np
import random

def ck_data(code):
	f = open('./finstate/'+code+'.txt','r')
	filedata = f.readlines()
	cnt = 0
	date_index = []
	for i in filedata:
		i = i.replace('\n','')
		filedata[cnt] = i
		if 'date' in i:
			date_index.append(cnt)
		cnt+=1


	date_index = random.sample(date_index,len(date_index))
	stock_f = open('./stockdata/'+code+'.txt','r')
	stock_data = stock_f.readlines()
	for i in date_index:
		date = filedata[i].replace('date : ','')+' 00:00:00'
		dt = datetime.strptime(date,'%Y-%m-%d %H:%M:%S').timetuple()
		timestamp = int(time.mktime(dt))
		if int(stock_data[0][:-1].split()[1]) >= timestamp or int(stock_data[len(stock_data)-1][:-1].split()[1]) <= timestamp+86400*61:
			continue
		cnt = 0
		bl = True
		for j in stock_data:
			j = j[:-1].split()
			if timestamp <= int(j[1]) and bl:
				start_index = cnt
				bl = False
			if(timestamp+86400*61 <= int(j[1])):
				change = int(stock_data[start_index][2]) - int(j[2])
				change_rate = change/int(stock_data[start_index][2])*100
				filedata = filedata[i+1,i+45]
				c = 0
				input_data = []
				for k in filedata:
					k = k.split()
					input_data.append(k[3])
					
				return True,change_rate,input_data

	
	return False,None







batch_size = 50

X = tf.placeholder(tf.float32,[None,44])
Y = tf.placeholder(tf.float32,[None,4])

e_step = tf.Variable(0,trainable=False, name='e_step')

L = tf.layers.dense(X,128,activation=tf.nn.relu)
L = tf.layers.dense(L,256,activation=tf.nn.relu)
L = tf.layers.dense(L,128,activation=tf.nn.relu)
L = tf.layers.dense(L,128,activation=tf.nn.relu)
output = tf.layers.ense(L,5,activation = None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=output))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./model')

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
	print('Model Restored!')
	saver.restore(sess,ckpt.model_checkpoint_path)
	print('e_step : ' ,sess.run(e_step))
else:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

f = open('./stockdata/code_list.txt','r')
code_list = f.readlines()
cnt = 0
for code in code_list:
	code = code[:-1]
	code_list[cnt] = code
	cnt+=1
f.close()

code_list = random.sample(code_list,len(code_list))
X_list = []
y_list = []
depth = 4
cnt = 0
while(cnt>=50):
	ck, change_rate,input_data = ck_data(code)
	x_list.append(input_data)
	if ck:
		cnt += 1
		if(change_rate > 20):
			y = 3
		elif(change_rate > 0):
			y = 2
		elif change_rate < 5:
			y = 1
		else:
			y = 0
		y_list = y.append() 

y_list = tf.one_host(y_list,depth)
episode_step = sess.run(e_step)

for episode_step in range(episode_step+1,episode_step+6):
	_,cost_val = sess.run([optimizer,cost],feed_dict={X:x_list,Y:y_list})
	print('Epoch : ', '%04d'%(epoch+1),'Avg.cost = ','{.3f}'.format(cost_val))


saver.save(sess,'./model/fs.ckpt')





	






