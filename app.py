from PIL import Image
import requests
import gradio as gr

from transformers import BlipProcessor, BlipForConditionalGeneration

model_id = "Salesforce/blip-image-captioning-base"

model = BlipForConditionalGeneration.from_pretrained(model_id)
processor = BlipProcessor.from_pretrained(model_id)

def launch(input):
    image = Image.open(requests.get(input, stream=True).raw).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

iface = gr.Interface(launch, inputs="text", outputs="text")
iface.launch()