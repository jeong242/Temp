import m_avg as m 

max = 0
for i in range(50000):
	new_acc = m.get_accuracy(m.weighted_moving_average,m.ETH_1,True,0,0.005*i)
	if new_acc > max:
		max = new_acc
		print("new max = "+str(max)+" with var = "+str(0.0005*i))
	if i % 1000 == 0:
		print(str(i)+"th iter")
