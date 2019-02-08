import numpy as np
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense 
import json

Num_epochs = 480
Batch_size = 160

with open('../../data/BTC_1.json', 'r') as r:
    BTC_1 = json.load(r)
 
Xtrain = np.empty(shape=(len(BTC_1),120,4))
Ytrain = np.zeros(shape=(len(BTC_1),3))

# 'i' for increase / 'c' for constant / 'd' for decrease
def delta(init, final):
    diff = float(final) - float(init)
    if diff > 0.0005:
        return "Rise"
    elif diff >= -0.0005:
        return "Steady"
    return "Fall"

for i in range(len(BTC_1)-210):
    for j in range(i,i+120):
        Xtrain[i,j-i,0] = float(BTC_1[j]['price_open']) / 1000
        Xtrain[i,j-i,1] = float(BTC_1[j]['price_close']) / 1000
        Xtrain[i,j-i,2] = float(BTC_1[j]['price_high']) / 1000
        Xtrain[i,j-i,3] = float(BTC_1[j]['price_low']) / 1000
    change = delta(BTC_1[i]['price_open'],BTC_1[i+210]['price_open'])
    if change=="Fall":
        Ytrain[i,0] = 1
    elif change=="Steady":
        Ytrain[i,1] = 1
    else:
        Ytrain[i,2] = 1


model = Sequential()
model.add(LSTM(240, input_shape=(120,4), go_backwards=True,
							 activation='relu',return_sequences=False))
model.add(Dense(3))

model.compile(loss='mean_squared_error', optimizer='adam',
							metrics=['mean_squared_error'])

train_size = len(Xtrain)//Batch_size * Batch_size

Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:train_size]

for i in range(Num_epochs):
	model.fit(Xtrain, Ytrain, batch_size=Batch_size, epochs=1,
						validation_split=0.2, verbose=1)
"""
model.fit(Xtrain, Ytrain, batch_size=Batch_size, epochs=Num_epochs,
						validation_split=0.2, verbose=2)
"""

score,_ = model.evaluate(Xtrain,Ytrain, batch_size=Batch_size, verbose=0)
rmse = math.sqrt(score)
print('\n MSE: {:.3f}, RMSE: {:.3f}'.format(score,rmse))
