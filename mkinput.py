import random
from datetime import datetime
import time
import numpy as np

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
			if timestamp+86400*61 <= int(j[1]):
				change =  int(j[2]) - int(stock_data[start_index][2])
				#print(stock_data[start_index][2])
				change_rate = change/int(stock_data[start_index][2])*100
				filedata = filedata[i+1:i+45]
				c = 0
				input_data = []
				for k in filedata:
					k = k.split()
					input_data.append(np.float64(k[3])/100000000)
					
				return True,change_rate,input_data
			stock_data[cnt] = j
			cnt+=1

	
	return False,None,None