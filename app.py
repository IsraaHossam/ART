
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
import numpy as np
import io

# Load the TensorFlow Hub model for style transfer
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
model = hub.load(hub_handle)

app = FastAPI()

# Function to preprocess the image
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = image[tf.newaxis, :]
    return image

@app.post("/apply_style/")
async def apply_style(content_image: UploadFile = File(...), style_image: UploadFile = File(...)):
    content_image_bytes = await content_image.read()
    style_image_bytes = await style_image.read()

    content_image_tensor = preprocess_image(content_image_bytes)
    style_image_tensor = preprocess_image(style_image_bytes)

    # Correct model call
    stylized_image = model(content_image=content_image_tensor, style_image=style_image_tensor)[0]

    stylized_image = np.array(stylized_image * 255, dtype=np.uint8)

    pil_image = Image.fromarray(stylized_image)

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return {
        "message": "Style transfer applied successfully",
        "image": img_byte_arr.getvalue()
    }
