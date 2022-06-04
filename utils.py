import base64
import io

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
from tensorflow import keras

print(tf.__version__)
print(tf.keras.__version__)

def get_model():
    global model
    model = load_model('model.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size,)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image