import numpy as np
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense 
import json

Num_epochs = 45
Batch_size = 160
Train_ratio = 0.7

# 'i' for increase / 'c' for constant / 'd' for decrease
def delta(init, final):
    diff = float(final) - float(init)
    if diff > 0.0005:
        return "Rise"
    elif diff >= -0.0005:
        return "Steady"
    return "Fall"

def data_preprocessing():
	with open('../../data/BTC_1.json', 'r') as r:
			BTC_1 = json.load(r)
 
	X = np.empty(shape=(len(BTC_1),120,4))
	Y = np.zeros(shape=(len(BTC_1),3))

	for i in range(len(BTC_1)-210):
			for j in range(i,i+120):
					X[i,j-i,0] = float(BTC_1[j]['price_open']) / 1000
					X[i,j-i,1] = float(BTC_1[j]['price_close']) / 1000
					X[i,j-i,2] = float(BTC_1[j]['price_high']) / 1000
					X[i,j-i,3] = float(BTC_1[j]['price_low']) / 1000
			change = delta(BTC_1[i]['price_open'],BTC_1[i+210]['price_open'])
			if change=="Fall":
					Y[i,0] = 1
			elif change=="Steady":
					Y[i,1] = 1
			else:
					Y[i,2] = 1

	train_size = int(Train_ratio*len(X)//Batch_size * Batch_size)
	Xtrain, Ytrain = X[0:train_size], Y[0:train_size]
	Xtest, Ytest = X[train_size+1:-1], Y[train_size+1:-1]

	return Xtrain, Ytrain, Xtest, Ytest

def model_building():
	model = Sequential()
	model.add(LSTM(240, input_shape=(120,4), go_backwards=True,
							 activation='relu',return_sequences=False))
	model.add(Dense(3))
	model.compile(loss='mean_squared_error', optimizer='adam',
							metrics=['mean_squared_error'])
	
	return model

def model_learning(model):
	for i in range(Num_epochs):
		model.fit(Xtrain, Ytrain, batch_size=Batch_size, epochs=1,
							validation_split=0.2, verbose=1)

# return the correct percentage.
# num_of_correct / num_of_total * 100
def compare(model, Xtest, Ytest):
	Xpredicted = model.predict(Xtest)
	count_correct = 0
	for i, predicted in enumerate(Xpredicted):
		predicted_value = np.argmax(predicted)
		true_value = np.argmax(Ytest[i])
		if predicted_value == true_value:
			count_correct += 1
	return count_correct / len(Xtest) * 100

if __name__ == "__main__":
	Xtrain, Ytrain, Xtest, Ytest = data_preprocessing()
	model = model_building()
	model_learning(model)
	model.save('lstm.h5')
	correct_ratio = compare(model, Xtest, Ytest)
	print("correct ratio = %d"%correct_ratio+"%")

"""
score,_ = model.evaluate(Xtrain,Ytrain, batch_size=Batch_size, verbose=0)
rmse = math.sqrt(score)
print('\n MSE: {:.3f}, RMSE: {:.3f}'.format(score,rmse))
"""
