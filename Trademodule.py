import urllib.request
import json
import datetime
import random

#(판매가격)
"""
url = "https://api.bithumb.com/public/ticker/BTC"
url_rest = "https://api.bithumb.com/public/orderbook/BTC"
url_past_data = "https://api.bithumb.com/public/recent_transactions/BTC"
html = urllib.request.urlopen(url)
html_rest = urllib.request.urlopen(url_rest)
api = html.read()
api_rest = html_rest.read()



js = json.loads(api)
js_rest = json.loads(api_rest)

data = js['data']
data_rest = js_rest['data']
timestamp = int(data['date'])
time = datetime.datetime.fromtimestamp(timestamp/1000)
"""


def get_sell_price():
    MAX_CUR = 3
    total_cur = 0
    total_price = 0
    for i in data_rest['asks']:
        q = float(i['quantity'])
        p = float(i['price'])
        if (MAX_CUR - (total_cur+q)) > 0:
            total_price += p*q
            total_cur += q
        else :
            total_price += (MAX_CUR-total_cur)*p
            break
    avg_price_cur = total_price/MAX_CUR
    return avg_price_cur

def get_all_data():
    f = open('./data/time_data.txt','r')
    data = f.readlines()
    data_dic = []
    data_len = len(data)
    for i in data:
        space_index = i.find(' ')
        time = i[0:space_index]
        price = i[space_index+1:-1]
        d = []
        d.append(time)
        d.append(price)
        data_dic.append(d)

    f.close()
    return data_len, data_dic

def get_all_data_timestamp():
    f = open('./data/timestamp_data.txt','r')
    data = f.readlines()
    data_dic = []
    data_len = len(data)
    for i in data:
        space_index = i.find(' ')
        timestamp = i[0:space_index]
        price = i[space_index+1:-2]
        d = []
        d.append(timestamp)
        d.append(price)
        data_dic.append(d)

    f.close()
    return data_len, data_dic

def get_startpoint(data_dic,input_size,period,BATCH_SIZE):
    for i in range(len(data_dic)):
        if int(data_dic[i][0]) > 1513223100 : #12월 15일 기준
            start_point = []
            for j in range(int(BATCH_SIZE/2)):
                start_point.append(random.randrange(input_size,i))
            for j in range(int(BATCH_SIZE/2+(BATCH_SIZE%2))):
                start_point.append(random.randrange(i,(len(data_dic)-period-1)))
            break

    return start_point














