import json

##### Different moving average functions #####
# Simple moving average
def simple_moving_average(start_index, list, Window_size):
	average = 0
	for i in range(start_index, start_index+Window_size):
		average += float(list[i]['price_close'])
	return average/Window_size

# Weighted with fib sequence
def weighted_moving_average(start_index, list, Window_size):
	import fib as F
	fib_list = F.fib_list(Window_size)
	average = 0
	j = 0
	for i in range(start_index, start_index+Window_size):
		average += fib_list[j]*float(list[i]['price_close'])
		j += 1 
	return average/sum(fib_list)

"""
{'price_high': 3584.11, 'volume_traded': 33.525391, 'time_period_end': '2019-01-13 08:01:00', 'price_close': 3579.47, 'price_open': 3584.1, 'time_period_start': '2019-01-13 08:00:00', 'price_low': 3579.47}
"""

# 'i' for increase / 'c' for constant / 'd' for decrease
def delta(init, final):
	if final > init:
		return 'i'
	elif final == init:
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
	total_iter = len(list)-Window_size
	for i in range(total_iter):
		init = float(list[i]['price_open'])
		final = float(list[i+Window_size]['price_close'])
		predicted = moving_average(i,list,Window_size)
		finals += [final]
		predicteds += [predicted]
	 	
		true_result = delta(init,final)
		predicted_result = delta(init,predicted)
		if true_result == predicted_result:
			count += 1
	return count, total_iter, finals, predicteds 

# Return lists of true final values & predicted values.
def for_plot(moving_average, list, Window_size):
	_, _, f, p = measure(moving_average, list, Window_size)
	return f,p

# Find the percentage of correct predictions.
def get_accuracy(moving_average, list, Window_size):
	result = measure(moving_average, list, Window_size)
	return (result[0]/result[1])*100
