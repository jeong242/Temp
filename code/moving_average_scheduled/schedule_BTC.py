#!/usr/bin/python3
from pymongo import MongoClient
import json
import get_data 
import m_avg
from datetime import datetime
from datetime import timedelta
from pytz import timezone

# KST timezone
kst = datetime.now(tz=timezone("Asia/Seoul")) + timedelta(hours=2)
prediction_time = kst.strftime("%Y-%m-%d %H:00:00")

# Either Rise / Steady / Fall
prediction = m_avg.measure(m_avg.simple_moving_average, get_data.BTC_1)

Client = MongoClient(host="13.125.150.105", port=27017, username="voteAdmin", password="voteAdmin")

result = {"H_nick_name":"moving_avg",
	  "H_model_name":"GoGoAI",
	  "H_Model_description":"moving_avg",
	  "H_pred_time":prediction_time,
	  "H_server_name":"GoGoAI_server",
	  "H_pred_movement":prediction}

DB_name = Client["GoGoAI"]
Collection_A = DB_name["BTC_results"]
Collection_B = DB_name["ETH_results"]

Collection_A.insert_one(result)
