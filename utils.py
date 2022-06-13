import base64
import os
from urllib import request

import cv2
import numpy as np
import requests
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
from tensorflow import keras

from decode_output import (BoundingBox, RescaleOutput,
                           calculate_nonmax_suppression, get_image_boxes)

######## OUTPUT CONFIGURATION ###########
LABELS = ['bicycle', 'bus', 'car', 'motorbike', 'person']
GRID_H, GRID_W = 13, 13
ANCHORS = np.array([0.07095013, 0.13790466, 0.74620075, 0.8126473, 0.37125614, 0.65841728, 0.18252735, 0.41417845])
ANCHORS[::2], ANCHORS[1::2] = ANCHORS[::2] * GRID_W, ANCHORS[1::2] * GRID_H
IMG_HEIGHT, IMG_WIDTH = 416, 416
GROUNDTRUTH_BOX = 20
obj_threshold = 0.1
iou_threshold = 0.4
cwd = os.getcwd()

model_dir = f'{cwd}/model.h5'
print(" * Model dir : {}".format(model_dir))

def get_model():
    global model
    print(' * Load model...')
    model = load_model(model_dir, compile=False)
    print(" * Model loaded!")
    return model

def load_image(url): 
    req = request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1) # 'Load it as it is'
    cv2.imwrite("image_loaded.jpeg", img)
    return img
    # res = requests.get(url)
    # if res == 200 and 'jpeg' in res.headers['content-type']:
    #     img_arr = np.array(Image.open(io.BytesIO(res.content)))
    #     img_arr = img_arr[:, :, ::-1].copy() 
    #     print(" * Image loaded : {}".format(img_arr))
    #     cv2.imwrite(url, img_arr)
    #     return img_arr
    # else: 
    #     return None

def predict_image(image):
    img = image
    result_image = None
    img_boxes = []
    img = cv2.imdecode(image, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    # cv2.imwrite("image.jpeg", img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    X = np.expand_dims(img, axis=0)
    Y = np.zeros((1, 1, 1, 1, GROUNDTRUTH_BOX, 4))
    
    output = model.predict([X, Y])
    # print(" * Output {}".format(output))
    rescaleResult = RescaleOutput(ANCHORS)
    rescaled_result = rescaleResult.fit(output[0])
    # print(" * Rescaled result {}".format(rescaled_result))
    img_boxes = get_image_boxes(rescaled_result, obj_threshold)
    # print(" * IMG Boxes {}".format(img_boxes))
    if img_boxes:
        img_boxes = calculate_nonmax_suppression(img_boxes, iou_threshold, obj_threshold)
        image = draw_boxes(X[0], img_boxes, LABELS)
        # cv2.imwrite("result.jpeg", image * 255.)
        result_image = image * 255.
        result_image = base64.encodebytes(result_image.tobytes()).decode('utf-8')
        # cv2.imwrite("result.jpeg", image)

    adjust_boxes = lambda n, nmax: max(min(nmax, n), 0)

    final_output = []
    for i in range(len(img_boxes)):
        final_output.append({
            'label': LABELS[img_boxes[i].label],
            'confidence': img_boxes[i].confidence,
            'xmin': adjust_boxes(int(img_boxes[i].x_min * IMG_WIDTH), IMG_WIDTH),
            'ymin': adjust_boxes(int(img_boxes[i].y_min * IMG_HEIGHT), IMG_HEIGHT),
            'xmax': adjust_boxes(int(img_boxes[i].x_max * IMG_WIDTH), IMG_WIDTH),
            'ymax': adjust_boxes(int(img_boxes[i].y_max * IMG_HEIGHT), IMG_HEIGHT)
        })
    return final_output, result_image


def draw_boxes(image, img_boxes, labels):
    image_h, image_w, _ = image.shape
    
    adjust_boxes = lambda n, nmax: max(min(nmax, n), 0)
    color_palette = list([tuple(np.random.choice(range(255), size=3) / 255.) for i in range(8)])
    for box, color in zip(img_boxes, color_palette):
        x_min = adjust_boxes(int(box.x_min * image_w), image_w)
        y_min = adjust_boxes(int(box.y_min * image_h), image_h)
        x_max = adjust_boxes(int(box.x_max * image_w), image_w)
        y_max = adjust_boxes(int(box.y_max * image_h), image_h)

        print(f'{labels[box.label]} {box.get_highest_label_probability_score() * 100}% [x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}]')
        cv2.rectangle(image,
                      pt1=(x_min,y_min), 
                      pt2=(x_max,y_max), 
                      color=color
                      )
        cv2.putText(img=image, 
                    text=f'{labels[box.label]} {int(box.get_highest_label_probability_score() * 100)}%', 
                    org=(x_min+ 13, y_min + 13),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1e-3 * image_h,
                    color=(1, 0, 1)
                    )     
    return image










######## PREDICT OUTPUT ##############
# input_image_path = './gambar.jpg'
#output_filename = 'gambar.jpg'


# image = cv2.imread(input_image_path)
# image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
# image = image / 255.

# X = np.expand_dims(image, axis=0)



