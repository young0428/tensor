#-*- coding: utf-8 -*-
import os
cp_index_list = []
def find_date_index(data):
        ct = 0
        for i in data:
            if(i.count('-') == 2):
                return ct
            ct += 1
for z in range(3,17):
    data_file_name = str(2015+int(z/4)) + '_' + str(z%4+1) + 'Q'
    f = open('./finstate/finstate_all/'+ data_file_name +'.txt','r')
    f.readline()
    f.readline()
    data_val = []
    word = ['유동자산', '현금및현금성자산', '매출채권', '재고자산', '기타유동금융자산', '기타유동자산', '매각예정비유동자산', '비유동자산', '장기금융상품', '기타포괄손익-공정가치', '당기손익인식공정가치측정금융자산', '종속회사투자', '관계회사투자', '렌탈자산', '유형자산', '무형자산', '기타비유동금융자산', '기타비유동자산', '사용권자산', '이연법인세자산', '자산총계', '유동부채', '단기매입채무', '단기차입금', '유동성장기차입금', '유동성사채', '기타유동금융부채', '기타유동부채', '당기법인세부채', '유동리스부채', '비유동부채', '사채', '장기차입금', '기타비유동금융부채', '이연법인세부채', '비유동리스부채', '부채총계', '자본금', '자본잉여금', '자본조정', '기타포괄손익누계액', '이익잉여금(결손금)', '자본총계', '자본과부채총계']
    wordlist = ['0']*44
    wordlist2 = ['0']*44
    code = ''
    code_val = '111'
    cnt = 0
    index = 0
    x1 = 0
    x2 = 0
    cp_name = ''
    a = 0

    line_num = len(f.readlines())
    f.seek(0)
    filepath = ''
    cp_val = [] 
    list_num = 0
    while(True):
        cnt = 0
        a += 1
        data = f.readline()
        if not(data):
            for i in range(44):
                fopen.write(cp_name + ' ' + code_val + ' ' + word[i] + ' ' + wordlist[i] + '\n')
            break
        data = data.split()
        for i in data:
            l = 0
            for j in word:
                if(i == j):
                    index = cnt
                    list_num = l
                    break
                l += 1
            if(index != 0 ):
                break
            cnt += 1
        
        if(index != 0):
            code = data[2].replace('[','').replace(']','')
            if(code_val != code):
                if(filepath != ''):
                    for i in range(44):
                        fopen.write(cp_name + ' ' + code_val + ' ' + word[i] + ' ' + wordlist[i] + ' ' + wordlist2[i] + '\n')
                    wordlist = ['0']*44
                    wordlist2 = ['0']*44
                    fopen.close()
                 
                filepath = './finstate/'+str(code)+'.txt'
                if(os.path.exists(filepath)):
                    option = 'a'
                else:
                    option = 'w'
                    cp_index_list.append(str(code))
                fopen = open(filepath,option)
                fopen.write('date : '+ data[find_date_index(data)] + '\n')
            if(len(data) >= index+1+1):
                x1 = data[index+1]
            if(len(data) >= index+1+2):
                x2 = data[index+2]
            wordlist[list_num] = x1
            wordlist2[list_num] = x2
            cp_name = data[3]
            x1 = '0'
            code_val = code
            index = 0

        if(a % 100 == 0):
            print(str(a)+'/'+str(line_num) + ' - ' + str(z-3) + '/' + str(14))

    f.close()
    fopen.close()
f = open('./finstate/code_list.txt','w')
for i in cp_index_list:
    f.write(i + '\n')
f.close()
        
                

    

