import copy
def aa(a):
	a[0] = 1


c = [4,4,4]
rate = copy.deepcopy(c)
aa(c)
print(rate)