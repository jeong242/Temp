import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense 
import pymongo
from pymongo import MongoClient


"""
{'_id': ObjectId('5c3f1217c30ba54698d8c0b8'), 'price_high': 3598.27, 'volume_traded': 30.971274, 'time_period_end': '2019-01-13 12:01:00', 'price_close': 3596.97, 'price_open': 3593.27, 'time_period_start': '2019-01-13 12:00:00', 'price_low': 3593.03}
"""

db = MongoClient('13.125.150.105',
								 	27017, 
									username='voteAdmin', 
									password='voteAdmin',
									authSource='BINANCE').BINANCE 
BTC_1 = list(db.get_collection('BTC_USD_1MIN').find({}))

Xtrain = np.empty(shape=(len(BTC_1),1,4))
Ytrain = np.empty(shape=(len(BTC_1),1,1))

for i in range(len(BTC_1)-1):
	Xtrain[i,0,0] = BTC_1[i]['price_open']
	Xtrain[i,0,1] = BTC_1[i]['price_close']
	Xtrain[i,0,2] = BTC_1[i]['price_high']
	Xtrain[i,0,3] = BTC_1[i]['price_low']

	Ytrain[i,0,0] = BTC_1[i+1]['price_open']

model = Sequential()
model.add(LSTM(259, input_shape=(1,4), go_backwards=True,
							 activation='relu',return_sequences=False))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam',
							metrics=['mean_squared_error'])

train_size = len(Xtrain)//Batch_size * Batch_size

Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:train_size]

for i in range(Num_epochs):
	model.fit(Xtrain, Ytrain, batch_size=Batch_size, epochs=1,
						validation_split=0.2, verbose=0)

score,_ = model.evaluate(Xtrain,Ytrain, batch_size=Batch_size, verbose=0)
rmse = math.sqrt(score)
print('\n MSE: {:.3f}, RMSE: {:.3f}'.format(score,rmse))
