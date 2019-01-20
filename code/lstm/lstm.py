import numpy as np
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense 
import json

Num_epochs = 480
Batch_size = 160

with open('../../data/BTC_1.json', 'r') as r:
	BTC_1 = json.load(r)
 
Xtrain = np.empty(shape=(len(BTC_1),1,4))
Ytrain = np.empty(shape=(len(BTC_1),1))

for i in range(len(BTC_1)-1):
	Xtrain[i,0,0] = BTC_1[i]['price_open']
	Xtrain[i,0,1] = BTC_1[i]['price_close']
	Xtrain[i,0,2] = BTC_1[i]['price_high']
	Xtrain[i,0,3] = BTC_1[i]['price_low']

	Ytrain[i,0] = BTC_1[i+1]['price_open']

model = Sequential()
model.add(LSTM(259, input_shape=(1,4), go_backwards=True,
							 activation='relu',return_sequences=False))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam',
							metrics=['mean_squared_error'])

train_size = len(Xtrain)//Batch_size * Batch_size

Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:train_size]

"""
for i in range(Num_epochs):
	model.fit(Xtrain, Ytrain, batch_size=Batch_size, epochs=1,
						validation_split=0.2, verbose=0)
"""
model.fit(Xtrain, Ytrain, batch_size=Batch_size, epochs=Num_epochs,
						validation_split=0.2, verbose=2)

score,_ = model.evaluate(Xtrain,Ytrain, batch_size=Batch_size, verbose=0)
rmse = math.sqrt(score)
print('\n MSE: {:.3f}, RMSE: {:.3f}'.format(score,rmse))
