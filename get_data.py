import pymongo
from pymongo import MongoClient

db = MongoClient('13.125.150.105',
								 	27017, 
									username='voteAdmin', 
									password='voteAdmin',
									authSource='BINANCE').BINANCE 
BTC_15 = list(db.get_collection('BTC_USD_15MIN').find({}))

