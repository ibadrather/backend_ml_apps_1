import requests
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}


def hf_translation_request(payload, API_URL):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


class Payload(BaseModel):
    text: str
    input_language: str
    output_language: str
