import random
from datetime import datetime
import time
import numpy as np
import copy
def mk_data(code):
	stock_f = open('./stockdata/'+code+'.txt','r')
	stock_data = stock_f.readlines()
	
	if len(stock_data)-73*182 < 0:
		return False,None,None

	n_able_index = len(stock_data)-73*182
	start_index = random.randrange(0,n_able_index)

	for i in range(80):
		if int(stock_data[start_index + i][:-1].split()[0])/100 == 900 :
			start_index = start_index + i
			break

	price_x = []
	price_y = []
	x_val = []
	day = 90
	for i in range(day):
		for j in range(13):
			x = stock_data[start_index + i*73 + j*6][:-1].split()[2]
			x_val.append(np.float32(x))
			if i < 30:
				y = stock_data[start_index + day*73 + i*73 + j*6][:-1].split()[2]
				price_y.append(np.float32(y))
			
		if (i+1)%30 == 0:
				price_x.append(x_val)
				x_val = []

	

	return True,price_x,price_y


def x_to_rate(x):
	rate = copy.deepcopy(x)
	for i in range(len(x)):
		base = x[i][0][0]
		for j in range(len(x[i])):
			for k in range(len(x[i][j])):
				rate[i][j][k] = ((x[i][j][k]/base)-1)*1000

	return rate

def y_to_rate(y,x):
	rate = copy.deepcopy(y)
	for i in range(len(y)):
		base = x[i][0][0]
		for j in range(len(y[i])):
			rate[i][j] = ((y[i][j]/base)-1)*1000
	return rate 

def rate_to_x(rate,x):
	res = copy.deepcopy(rate)
	for i in range(len(rate)):
		base = x[i][0][0]
		for j in range(len(rate[i])):
			res[i][j] = base*((rate[i][j]/1000+1))

	return res








	