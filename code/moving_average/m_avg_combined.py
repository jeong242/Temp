import json
every = 240 
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
	average = float(list[0]) * multiplier

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
with open('../../data/ETH_1.json', 'r') as r:
	ETH_1 = json.load(r)

##### Prediction using moving average #####
# Return number of correct predictions.
def measure_combined(list, Window_size=0, normal = True, mu=0, sd=1,probability=0):
	from numpy import random
	count = 0
	finals = []
	predicteds = []
	bound = len(list)-211
	total_iter = 0

	for i in range(0,bound,every):
		# initial value of the sample data
		init = float(list[i+90]['price_close'])
		# final value of the sample data
		final = float(list[i+211]['price_open'])

		# Predicting using simple moving average method
		# "known" sample data for predicting Window_size later
		known = [list[n]['price_open'] for n in range(i,i+120)]	
		for j in range(90):
			predicted = simple_moving_average(0,known,120+j)
			if normal:
				known += [predicted+random.normal(mu,sd)]
			else:
				known += [predicted]
		predicted_simple = predicted 

		# Predicting using weight moving average method
		# "known" sample data for predicting Window_size later
		known = [list[n]['price_open'] for n in range(i,i+120)]	
		for j in range(90):
			predicted = weighted_moving_average(0,known,120+j)
			if normal:
				known += [predicted+random.normal(mu,sd)]
			else:
				known += [predicted]
		predicted_weight = predicted 

		# Predicting using alpha moving average method
		# "known" sample data for predicting Window_size later
		known = [list[n]['price_open'] for n in range(i,i+120)]	
		for j in range(90):
			predicted = alpha_moving_average(0,known,120+j)
			if normal:
				known += [predicted+random.normal(mu,sd)]
			else:
				known += [predicted]
		predicted_alpha = predicted 

		predicted_result_simple = delta(init,predicted_simple)
		predicted_result_weight = delta(init,predicted_weight)
		predicted_result_alpha  = delta(init,predicted_alpha)
		# count number of inc/dec/constant
		num_of_inc = num_of_dec = num_of_con = 0
		
		if predicted_result_simple == 'i':
			num_of_inc += 1
		if predicted_result_weight == 'i':
			num_of_inc += 1
		if predicted_result_alpha == 'i':
			num_of_inc += 1

		if predicted_result_simple == 'd':
			num_of_dec += 1
		if predicted_result_weight == 'd':
			num_of_dec += 1
		if predicted_result_alpha == 'd':
			num_of_dec += 1

		if predicted_result_simple == 'c':
			num_of_con += 1
		if predicted_result_weight == 'c':
			num_of_con += 1
		if predicted_result_alpha == 'c':
			num_of_con += 1
		
		if num_of_inc - num_of_dec == 1:
			predicted_result = 'i'
		elif num_of_dec - num_of_inc == 1:
			predicted_result = 'd'
		elif num_of_inc == 3:
			predicted_result = 'i'
		elif num_of_dec == 3:
			predicted_result = 'd'
		elif num_of_inc == 1 and num_of_dec == 1 and num_of_con == 1:
			prob = randint(0,10000)
			if prob < 4500:
				predicted_result = 'i'
			elif prob < 9000:
				predicted_result = 'd'
			else:
				predicted_result = 'c'
		else:
			predicted_result = predicted_result_weight

		true_result = delta(init,final)
		
		from random import randint
		prob = randint(0,1000)
		if prob < probability:
			if predicted_result == 'i':
				predicted_result = 'd'
			else:
				predicted_result = 'i'
		
		if true_result == predicted_result:
			count += 1

		total_iter += 1

	return count, total_iter 

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
def get_accuracy_combined(list, normal=True,mu=0,sd=1,probability=0):
	result = measure_combined(list,0,normal,mu,sd,probability)
	return (result[0]/result[1])*100
