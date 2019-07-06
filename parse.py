import urllib.request
import json
import time
from datetime import datetime


"""https://finance.daum.net/api/charts/D0011001/5/minutes?limit=200&adjusted=true&to=2019-02-11%2011%3A30%3A00.0""" #형식
f = open('./data/time_data.txt','w')

url = "https://finance.daum.net/api/charts/A005930/5/minutes?limit=216&adjusted=true&to=2016-06-04%2011%3A30%3A00.0"
hdr = {'referer' : 'https://finance.daum.net/domestic/chart', 'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}
req = urllib.request.Request(url,headers=hdr)
res = urllib.request.urlopen(req)
#print(res.read())
js = json.loads(res.read())
text = ''
while(len(js['data']) == 216):
	string = ''
	for i in js['data']:
		s = i['candleTime'].replace('.0','')
		time_time = s.replace(':','').replace(' ','').replace('-','')
		print(time_time)
		print(i)
		dt = datetime.strptime(s,'%Y-%m-%d %H:%M:%S').timetuple()
		timestamp = time.mktime(dt)
		#print(timestamp)
		string = string + str(time_time) + ' ' + str(int(timestamp)) + ' ' + str(int(i['tradePrice'])) + '\n'

	text = string + text
	s = js['data'][0]['candleTime'].replace('.0','')
	time_time = s.replace(':','').replace(' ','').replace('-','')
	year = str(int(int(time_time)/(10**10)))
	month = str(int((int(time_time)%(10**10))/(10**8))).zfill(2)
	day = str(int((int(time_time)%(10**8))/(10**6))).zfill(2)
	print(year,month,day)
	url = "https://finance.daum.net/api/charts/A005930/5/minutes?limit=216&adjusted=true&to=" + year + "-" + month + "-" + day + "%2011%3A30%3A00.0"
	req = urllib.request.Request(url,headers=hdr)
	res = urllib.request.urlopen(req)
	js = json.loads(res.read())

f.write(text)
f.close()
print(len(js['data']))

	





