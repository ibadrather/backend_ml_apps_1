import requests
from pydantic import BaseModel
from dotenv import load_dotenv
import os


def config():
    load_dotenv()


config()

API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-nl"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}


def en_nl_query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


class Payload(BaseModel):
    text: str
