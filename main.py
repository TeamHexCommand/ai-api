from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import urllib.request
import os
import openai
from dotenv import load_dotenv
import cv2
import numpy as np
from transparent_background import Remover
import uuid 

load_dotenv()

model_id = "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained(model_id)
processor = BlipProcessor.from_pretrained(model_id)

openai.api_key = os.getenv("OPENAI")
model_engine = "text-davinci-003"

remover = Remover()

output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def blipCaption(img: str):
    image = Image.open(requests.get(img, stream=True).raw).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def removeBackground(img: str):
    image = Image.open(requests.get(img, stream=True).raw).convert('RGB')
    out = remover.process(image, type='rgba')
    name = uuid.uuid4()
    filename = f'{output_folder}/{name}.png'
    out.save(filename)
    return name

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "BLIP Running"}

@app.get("/info")
def read_info():
    return {"msg": "Team HexCommand"}

@app.get("/blip/caption/{img}")
def read_item(img: str):
    ans = blipCaption(requests.utils.unquote(img))
    return {"img": requests.utils.unquote(img), "caption": ans}

@app.get("/openai/question/{prompt}")
def openai_questions(prompt: str):
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return {"prompt": prompt, "answer": completion.choices}


@app.get("/image/removebg/{img}")
def image_removebg(img: str):
    ans = removeBackground(requests.utils.unquote(img))
    image_url = get_image_url(ans)
    if image_url:
        return {"img": image_url}
    else:
        return {"img": uuid, "error": "Image not found"}


def get_image_url(uuid: str):
    output_folder = 'output'
    filename = f'{output_folder}/{uuid}.png'
    if os.path.exists(filename):
        return f'{os.getenv("BASEURL")}/{filename}'
    else:
        return None


@app.get("/image/{img}")
def read_image(img: str):
    uuid = requests.utils.unquote(img)
    image_url = get_image_url(uuid)
    
    if image_url:
        return FileResponse(image_url, media_type='image/png')
    else:
        return {"img": uuid, "error": "Image not found"}