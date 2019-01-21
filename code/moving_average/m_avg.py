import json
alpha = 0.5

##### Different moving average functions #####
# Simple moving average
def simple_moving_average(start_index, list, Window_size):
	average = 0
	for i in range(start_index, start_index+Window_size):
		average += float(list[i])
	return average/Window_size

# Weighted with fib sequence
def weighted_moving_average(start_index, list, Window_size):
	import fib as F
	fib_list = F.fib_list(Window_size)
	average = 0
	j = 0
	for i in range(start_index, start_index+Window_size):
		average += fib_list[j]*float(list[i])
		j += 1 
	return average/sum(fib_list)

def alpha_moving_average(start_index, list, Window_size):
	multiplier = alpha
	average = list[0] * multiplier

	for i in range(start_index+1, start_index+Window_size):
		multiplier *= (1-alpha)
		average += (float(list[i]) * multiplier) 
	return average

"""
{'price_high': 3584.11, 'volume_traded': 33.525391, 'time_period_end': '2019-01-13 08:01:00', 'price_close': 3579.47, 'price_open': 3584.1, 'time_period_start': '2019-01-13 08:00:00', 'price_low': 3579.47}
"""

# 'i' for increase / 'c' for constant / 'd' for decrease
def delta(init, final):
	diff = final - init
	if diff > 0.0005:
		return 'i'
	elif diff >= -0.0005:
		return 'c'
	return 'd'

# Load sample data
with open('../../data/BTC_1.json', 'r') as r:
	BTC_1 = json.load(r)

##### Prediction using moving average #####
# Return number of correct predictions.
def measure(moving_average, list, Window_size):
	count = 0
	finals = []
	predicteds = []
	total_iter = len(list)-2*Window_size

	for i in range(total_iter):
		# initial value of the sample data
		init = float(list[i+Window_size-1]['price_close'])
		# final value of the sample data
		final = float(list[i+2*Window_size]['price_open'])
		# "known" sample data for predicting Window_size later
		known = [list[n]['price_open'] for n in range(i,i+Window_size)]	

		# Predicting using moving average method
		for j in range(Window_size):
			predicted = moving_average(0,known,Window_size+j)
			known += [predicted]
		
		# Update the lists
		finals += [final]
		predicteds += [predicted]
		
		true_result = delta(init,final)
		predicted_result = delta(init,predicted)
		if true_result == predicted_result:
			count += 1

		print('Initial time: ' + list[i+Window_size-1]['time_period_end'])
		print('Final   time: ' + list[i+2*Window_size]['time_period_start']+'\n')
	return count, total_iter, finals, predicteds 

# Return lists of true final values & predicted values.
def for_plot(moving_average, list, Window_size):
	_, _, f, p = measure(moving_average, list, Window_size)
	return f,p

# MAE
def mae(moving_average, list, Window_size):
	f,p = for_plot(moving_average, list, Window_size)
	import numpy as np
	f = np.array(f)
	p = np.array(p)
	return  sum(np.abs(f-p)) / len(f)

# RMSE
def rmse(moving_average, list, Window_size):
	f,p = for_plot(moving_average, list, Window_size)
	import numpy as np
	import math
	f = np.array(f)
	p = np.array(p)
	return  math.sqrt( sum((f-p) ** 2)/len(f) )
	

# Find the percentage of correct predictions.
def get_accuracy(moving_average, list, Window_size):
	result = measure(moving_average, list, Window_size)
	return (result[0]/result[1])*100
