import urllib.request
import json
import datetime

url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1496275200&end=9999999999&period=300'
api = urllib.request.urlopen(url).read()

js = json.loads(api)
f = open('./data/timestamp_data.txt','w')
f_2 = open('./data/time_data.txt','w')

for i in js:
	timestamp = i['date']
	time = datetime.datetime.fromtimestamp(timestamp)
	time = str(time).replace('-','')
	time = time.replace(' ','')
	time = time.replace(':','')
	time = time[:-2]

	avg_price = i['weightedAverage']
	string = str(timestamp) + ' ' + str(avg_price) + '\n'
	f.write(string)
	string = time + ' ' + str(avg_price) + '\n'
	f_2.write(string)

f.close()
f_2.close()