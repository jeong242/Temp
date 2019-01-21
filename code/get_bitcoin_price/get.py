import requests
bitcoin_api_url = 'https://api.coindesk.com/v1/bpi/currentprice.json'

def get():
	response = requests.get(bitcoin_api_url)
	return response.json()
