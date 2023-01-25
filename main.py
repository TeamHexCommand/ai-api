from typing import Union
from fastapi import FastAPI
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import urllib.request
import os
import openai
from dotenv import load_dotenv

load_dotenv()

model_id = "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained(model_id)
processor = BlipProcessor.from_pretrained(model_id)

openai.api_key = os.getenv("OPENAI")
model_engine = "text-davinci-003"

def blipCaption(img: str):
    image = Image.open(requests.get(img, stream=True).raw).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "BLIP Running"}

@app.get("/info")
def read_root():
    return {"msg": "Team HexCommand"}

@app.get("/blip/caption/{img}")
def read_item(img: str):
    ans = blipCaption(requests.utils.unquote(img))
    return {"img": requests.utils.unquote(img), "caption": ans}

@app.get("/openai/question/{prompt}")
def read_item(prompt: str):
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return {"prompt": prompt, "answer": completion.choices}
