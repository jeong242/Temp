"""
	Get sample data then save as JSON format
"""
import pymongo
import json
from pymongo import MongoClient

db = MongoClient('13.125.150.105',
								 	27017, 
									username='voteAdmin', 
									password='voteAdmin',
									authSource='BINANCE').BINANCE 

BTC_1_temp  = list(db.get_collection('BTC_USD_1MIN').find({}))

BTC_15_temp = list(db.get_collection('BTC_USD_15MIN').find({}))
BTC_30_temp  = list(db.get_collection('BTC_USD_30MIN').find({}))

# Remove ObjectIDs since they're not serializable and I don't need them.
def remove_objectID(dict):
	del dict['_id']
	return dict 

BTC_1 = []
BTC_15 = []
BTC_30 = []


for dict in BTC_1_temp:
	del dict['_id']
	BTC_1 += [dict] 
for dict in BTC_15_temp:
	del dict['_id']
	BTC_15 += [dict]
for dict in BTC_30_temp:
	del dict['_id']
	BTC_30 += [dict]

with open('BTC_1.json', 'w') as f:
	json.dump(BTC_1, f)
with open('BTC_15.json', 'w') as f:
	json.dump(BTC_15, f)
with open('BTC_30.json', 'w') as f:
	json.dump(BTC_30, f) 
