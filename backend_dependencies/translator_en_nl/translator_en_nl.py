import requests
from pydantic import BaseModel

API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-nl"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def en_nl_query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
class Payload(BaseModel):
    text: str
