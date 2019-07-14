import os
f = open('./finstate/code_list.txt','r')
f_2 = open('./finstate/code_list2.txt','w')
codelist = f.readlines()
print(len(codelist))
cnt = 0
for code in codelist:
	code = code[:-1]
	if os.path.exists('./stockdata/'+code+'.txt'):
		if os.path.getsize('./stockdata/'+code+'.txt') < 100:
			print('Delete :',code)
			os.remove('./stockdata/'+code+'.txt')
			code = ''
		else:
			f_2.write(code+'\n')
			cnt+=1
print(cnt)
f.close()
f_2.close()

