from moving_average import for_plot
import matplotlib.pyplot as plt
import numpy as np

def plot(moving_average, list, Window_size):
	finals, predicteds = for_plot(moving_average, list, Window_size)
	x = np.array([i for i in range(len(finals))])
	y_finals = np.array(finals)
	y_predicteds = np.array(predicteds)

	plt.plot(x, y_finals-y_predicteds, label = 'true')
	
	#plt.plot(x, y_finals, label = 'true')
	#plt.plot(x, y_predicteds, linestyle='--', label = 'predicted')
	plt.xlabel('time')
	plt.ylabel('price')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	import moving_average as M
	plot(M.weighted_moving_average, M.BTC_1, 120)
