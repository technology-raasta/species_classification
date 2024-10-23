import requests
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
import re
import os

# Initialize FastAPI app
app = FastAPI()

# Your class mapping dictionary
data_class = {'aloevera': 0,
              'banana': 1,
              'bilimbi': 2,
              'cantaloupe': 3,
              'cassava': 4,
              'coconut': 5,
              'corn': 6,
              'cucumber': 7,
              'curcuma': 8,
              'eggplant': 9,
              'galangal': 10,
              'ginger': 11,
              'guava': 12,
              'kale': 13,
              'longbeans': 14,
              'mango': 15,
              'melon': 16,
              'orange': 17,
              'paddy': 18,
              'papaya': 19,
              'peper chili': 20,
              'pineapple': 21,
              'pomelo': 22,
              'shallot': 23,
              'soybeans': 24,
              'spinach': 25,
              'sweet potatoes': 26,
              'tobacco': 27,
              'waterapple': 28,
              'watermelon': 29}

# Downloading Model


def download_file_from_google_drive(destination):
    if os.path.exists(destination):
        print(f"File already exists at {destination}")
        return

    URL = f"https://drive.usercontent.google.com/download?id=1-SvD9PxjLS5UrX87HAsdbDfG2B-qK_JD&export=download&authuser=0&confirm=t&uuid=f1082cff-7016-4074-9177-be7b69e0d6e6&at=AN_67v1rQ-jYokAHeX39y6bXB5E8%3A1729669285206"

    response = requests.get(URL)

    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully to {destination}")
    else:
        print(f"Failed to download file: {response.status_code}")


class ImageDataURI(BaseModel):
    data_uri: str


def load_and_preprocess_image(image_data):
    img = Image.open(BytesIO(image_data))
    img = img.resize((150, 150))  # Resize to match model input size
    img_array = np.array(img)
    # Shape becomes (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


download_file_from_google_drive("./model/eye.h5")
model = load_model("./model/eye.h5")


def decode_data_uri(data_uri):
    # Remove the "data:image/png;base64," prefix or other similar prefixes
    header, base64_str = data_uri.split(',', 1)

    # Decode the base64 string
    image_data = base64.b64decode(base64_str)
    return image_data


if __name__ == "__main__":

    @app.post("/predict/")
    async def predict(image_data_uri: ImageDataURI):
        # Decode the image from the Data URI
        image_data = decode_data_uri(image_data_uri.data_uri)

        # Preprocess the image
        img = load_and_preprocess_image(image_data)

        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Map prediction to class name
        predicted_label = None
        confidence = None
        for key, value in data_class.items():
            if value == predicted_class:
                predicted_label = key
                confidence = prediction[0][predicted_class]
                break

        return {"class": predicted_label, "confidence": float(confidence)}

# To run the API, use:
# uvicorn filename:app --reload
