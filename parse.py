import urllib.request
import json
import time
from datetime import datetime


"""https://finance.daum.net/api/charts/D0011001/5/minutes?limit=200&adjusted=true&to=2019-02-11%2011%3A30%3A00.0""" #형식
codefile = open('./stockdata/code_list.txt')
code_list = codefile.readlines()
cnt = 0
save = 1515
for code in code_list:
	cnt+=1
	if(cnt < save):
		continue

	print(str(cnt) + '/' + str(len(code_list)))
	code = str(code)
	code = code[ : -1]
	print(code)

	f = open('./stockdata/' + code + '.txt','w')

	url = "https://finance.daum.net/api/charts/A" + code + "/5/minutes?limit=4320&adjusted=true&to=2019-04-20%2011%3A30%3A00.0"
	hdr = {'referer' : 'https://finance.daum.net/domestic/chart', 'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}
	req = urllib.request.Request(url,headers=hdr)
	res = urllib.request.urlopen(req)
	js = json.loads(res.read())
	text = ''


	while(len(js['data']) == 4320):
		string = ''
		string_val = ''
		check = False
		for i in js['data']:
			s = i['candleTime'].replace('.0','')
			time_time = s.replace(':','').replace(' ','').replace('-','')
			if(int(int(int(time_time)%(10**6))/10**2) == 900):
				check = True
			
			if(check):
				dt = datetime.strptime(s,'%Y-%m-%d %H:%M:%S').timetuple()
				timestamp = time.mktime(dt)
				#print(timestamp)
				string_val = string_val + str(time_time) + ' ' + str(int(timestamp)) + ' ' + str(int(i['tradePrice'])) + '\n'

			if(int(int(int(time_time)%(10**6))/10**2) == 1500):
				string = string + string_val
				string_val = ''
				check = False

		text = string + text
		s = js['data'][0]['candleTime'].replace('.0','')
		dt = datetime.strptime(s,'%Y-%m-%d %H:%M:%S').timetuple()
		timestamp = time.mktime(dt)
		timestamp += 86400
		d = datetime.fromtimestamp(timestamp)
		time_time = str(d).replace(':','').replace(' ','').replace('-','')
		year = str(int(int(time_time)/(10**10)))
		month = str(int((int(time_time)%(10**10))/(10**8))).zfill(2)
		day = str(int((int(time_time)%(10**8))/(10**6))).zfill(2)



	# 타임스탬프로 바꾸고 1일 빼고 다시 날짜로 바꾸기 
		url = "https://finance.daum.net/api/charts/A" + code + "/5/minutes?limit=4320&adjusted=true&to=" + year + "-" + month + "-" + day + "%2011%3A30%3A00.0"
		req = urllib.request.Request(url,headers=hdr)
		res = urllib.request.urlopen(req)
		js = json.loads(res.read())

	f.write(text)
	f.close()

codefile.close()

	





