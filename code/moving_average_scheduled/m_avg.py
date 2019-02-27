import json
alpha = 0.033499

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
		return "Rise"
	elif diff >= -0.0005:
		return "Steady"
	return "Fall"

##### Prediction using moving average #####
# Return number of correct predictions.
def measure(moving_average, list):
	from numpy import random 
	# length of list of bitcoin prices.
	size = len(list)
	# initial value of the sample data
	init = float(list[89]['price_close'])
	# "known" sample data for predicting Window_size later
	known = [list[n]['price_open'] for n in range(size)]	

	print("size = "+str(size))

	# Predicting using moving average method
	for j in range(size-30):
		predicted = moving_average(0,known,size+j)
		known += [predicted + random.normal(0,0.02)]
		print(predicted)

	return delta(init,predicted)

"""
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
"""
