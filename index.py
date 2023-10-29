import uvicorn
from fastapi import FastAPI, File, UploadFile
import cv2
import tensorflow as tf

import torch
from PIL import Image
import io
from io import BytesIO
import numpy as np






app = FastAPI()
model = tf.keras.models.load_model("model")

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}


@app.post('/predict')
async def predict(file: UploadFile):
    # Read the uploaded image file
    image_data = await file.read()

    # Convert image data to a NumPy array
    np_image = np.frombuffer(image_data, np.uint8)

    # Decode the image using OpenCV
    test_img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Resize the image to 224x224
    test_img = cv2.resize(test_img, (224, 224))

    # Reshape the image for the model
    test_input = test_img.reshape((1, 224, 224, 3))

    # Make predictions
    prediction = model.predict(test_input)

    # Post-process the prediction
    if prediction[0] > 0.7:
        prediction_result = "Non-Infected"
    else:
        prediction_result = "Go and consult doctor"

    return {
        'prediction': prediction_result
    }







