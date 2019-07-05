#-*- coding: utf-8 -*-

f = open('./finstate/2019_1분기보고서_01_재무상태표_20190625.txt','r')
f.readline()
f.readline()
data_val = []
word = ['유동자산', '현금및현금성자산', '매출채권', '재고자산', '기타유동금융자산', '기타유동자산', '매각예정비유동자산', '비유동자산', '장기금융상품', '기타포괄손익-공정가치', '당기손익인식공정가치측정금융자산', '종속회사투자', '관계회사투자', '렌탈자산', '유형자산', '무형자산', '기타비유동금융자산', '기타비유동자산', '사용권자산', '이연법인세자산', '자산총계', '유동부채', '단기매입채무', '단기차입금', '유동성장기차입금', '유동성사채', '기타유동금융부채', '기타유동부채', '당기법인세부채', '유동리스부채', '비유동부채', '사채', '장기차입금', '기타비유동금융부채', '이연법인세부채', '비유동리스부채', '부채총계', '자본금', '자본잉여금', '자본조정', '기타포괄손익누계액', '이익잉여금(결손금)', '자본총계', '자본과부채총계']
code = ''
code_val = ''
cnt = 0
index = 0
x1 = 0
x2 = 0
cp_val = []
for k in range(700):
    cnt = 0
    data = f.readline().split()
    for i in data:
        for j in word:
            if(i == j):
                index = cnt
                break
        if(index != 0 ):
            break
        cnt += 1
    
    if(index != 0):
        code = data[2].replace('[','').replace(']','')
        if(code_val != code):
            cp_val.append(data_val)
            data_val = []
        if(len(data) >= index+1+1):
            x1 = data[index+1]
        if(len(data) >= index+1+2):
            x2 = data[index+2]
        data_val.append([data[3],code,data[index],x1,x2])
        x1 = 0
        x2 = 0
        code_val = code
        index = 0
    
        
                
for i in cp_val:
    for j in i:
        print(j)
    

