import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import base64
from urllib import request

import cv2
import flask
import gcsfs
import h5py
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
from tensorflow import keras
from werkzeug.datastructures import ImmutableMultiDict

from decode_output import (BoundingBox, RescaleOutput,
                           calculate_nonmax_suppression, get_image_boxes)

# from utils import get_model, load_image, predict_image

######## OUTPUT CONFIGURATION ###########
LABELS = ['bicycle', 'bus', 'car', 'motorbike', 'person']
GRID_H, GRID_W = 13, 13
ANCHORS = np.array([0.1826925422137672, 0.41441697626638907, 0.07093507577988668, 0.13790057307749862, 0.3710890099560204, 0.6587015133970981, 0.7463448677408963, 0.8127615496823053])
ANCHORS[::2], ANCHORS[1::2] = ANCHORS[::2] * GRID_W, ANCHORS[1::2] * GRID_H
IMG_HEIGHT, IMG_WIDTH = 416, 416
GROUNDTRUTH_BOX = 20
obj_threshold = 0.3
iou_threshold = 0.4
cwd = os.getcwd()

def get_model():
    global model
    print(' * Load model...')
    model = load_model('model.h5', compile=False)
    print(" * Model loaded!")
    return model

def load_model_from_bucket():
    global model
    print(' * Load model from bucket ...')
    PROJECT_NAME = 'securicam'
    # CREDENTIALS = "ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAic2VjdXJpY2FtLTM1MTkwNiIsCiAgInByaXZhdGVfa2V5X2lkIjogIjU3ZmY3MzBiNGM1ODRkMjIwYjRjYzVjYTBjNjA0YTg0OWE5NmQ2YjkiLAogICJwcml2YXRlX2tleSI6ICItLS0tLUJFR0lOIFBSSVZBVEUgS0VZLS0tLS1cbk1JSUV2QUlCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQktZd2dnU2lBZ0VBQW9JQkFRQzZFK3VrOE5lcTk5Tm1cbjlJQVZQdmhtOElWSVExaTJoT2Rjd094TzdLV0lpMXcvY1NJaE5FVkVla2I3VDUwYzdsWmYwQUxYNG1iWFlBcEJcblZWVnV0ajdvNmNJa0JqTzV4ejJ0d2J1NCtQVjRhb2JwUzYwTWlqL1A5Q3dmWFdtUGEzODJBT0JWcEV5dk5WcDRcbnk2TjZDZVlPZ0NvZWxWek54NmpRditxUXBMdE9yc0xEaFh4THJmdFh5eFFBbHBERzRGNTBMKzJoejUwaVlyTDlcbmtkeG9SSDVmNVBld21xTGc4TkRTK1dSSkZ1QVNZQng5blBsZUMvK1R0VUlaZndMMUl2YlpNd3JaTGJMd0tpWE1cbnJLUm5za0hBczFXMGVLSHV2ZU9FTHFuSlJ3L2N1UnhMMWlUL3dDTGF2cERJanl5S1EzWVhpckF2S1JTeEgxUWVcbnZrcVFSNkZYQWdNQkFBRUNnZ0VBQXcybTlmTXdwNm0wS29JTElSeFhkM2E3clpoaFQxWFpnNWtzMGlrcUNtdWFcbkcyZ3NjcFhQN3ltT1JTemRwbWNNV0pDeFpzNTQyOTdnTGZ0RXdzY0lkSzNYRG1xYStGanZqbE85VkRqNFFVeE5cbnY5NG5NWjBZUHhOQjZCYjF6Nk1vaWticzJUa2g1bzNodkhIT0dVNS9mMHhxaUJKL1A5ajBzK3BzRFZBYzUySDdcbmNxUGUvdnJwalBCYUN4cWgwbnBQVFM0cG1Ob0pyZ2I1UnZiekxUSjNteUtXaVdCSm5OYzQxdkRQazVPNjlCdVZcbmtwYkpLOGZNMXgwNm1qUEpVbjZjc294VFAxZXBBQVBlWDVCbkM2YXplY0RoRWxUVGU3NTk1akpaZTluSlpJYXVcbmdLL0VkalE3a1VNU3lMV0hZRm9UaldvWlBpL3MzRWNhZkV5L2FjcmxtUUtCZ1FEckc5MG5INWw1ZkVPb3JqL2tcbjdBSzNSbDAxMStET1BDZ0RFN2RDNm9ySXBVNFYveDlFVkg3UHZJL09BUXBod1VOc2psUUo5SHprOUQ4enc0Ym9cbk5xeTB0SGNYRGlkY3REZGk5Y2czdGIwdGlWMmlnRVFwajFKNjVsUEUrL1BoSEgyOS9ydWRHcUQ3MTY2SHgxbHVcbmZRRlVzSHhxR1lnRjZIYVN2dGRBSjYvYW53S0JnUURLbkxsSURHMTNQeWF4eFArMDhPRjg3VWVRQkd3Vk9vZWNcbm15MFQ4cVE5UnFNN2xJaStIM3VRRkVsOFJJbUh3aTJ0cUl6MDRwQW80d2FBdktMbTVvWUE1N2lUWGpRVW9wdGxcblJpU2V1ZnF1eHN2b291LzMyTXNFNUx4YWY0eWd0NTRxdWxVUjVMZmtRS0hPTkZTR0JNUUpzU1dodE9tNEJKZFRcbitQRVlENE4yU1FLQmdCdEhYRjJSdVFqemRoWTRRWW1hS0thREw0Vnp5czJqQlRGMk5DazdkV2JhWnpqK0pPNlVcbnJ6SURPdHc0R3JCRThFSEVNZFFGM2dmaW53S3VpUlNnWTJHRUh0MDU4eTg2YWNLOXFjRnRBTW9yeWdWMHhIUGVcbkFUL3BRWWIxaE5KMXI5RS9vUjBWWGVCYW9oRjIrOVZxQTZHRDZLdVcwc1Y4dDJWSFowbzJVSjBaQW9HQUxkOXBcbis2d0VwTGtCazhJY2V1VUd5SGdZZWJ6bWs2L2F3RlJKMG5oZlF4aFpJTVl5WjRsTk9vTzlWNHRVOEEzQXRjNnhcbmkwZzRoMmxQTVpxRDcyOUY2N0tMRWFLRWZCK011MU0wTzFPME5Cb0NWTHQvUlVncVB3Tml1Y0tqSGtnMFlVd3FcbnNwQnNLaHVRQnRYR1JVbVM4UGJRcEZvSGlJaFJrc3VNR254NHNURUNnWUF5UHI4SnNyZC9STnhsc2N5dFhCMFdcblJXVUdCOXFNZHhwSEhoMnVLd3ZXQ1ArbHlCWFBUV1YybFlIWjlZRnprK1UrQWxXalBBaTJnYzBmbTdDcytuUzNcbmx1KzcrQW94aTVwczc3ckgwdHZhdUFqTXNoVk5PYXFaZ25BOUczNWJrbkZSZUxQV1ZHRjVRelBZeDY3ajJUMkpcblRHaVgvU1p2Z1ZmcXZiekVZd0NCVlE9PVxuLS0tLS1FTkQgUFJJVkFURSBLRVktLS0tLVxuIiwKICAiY2xpZW50X2VtYWlsIjogInNlY3VyaWNhbS0zNTE5MDZAYXBwc3BvdC5nc2VydmljZWFjY291bnQuY29tIiwKICAiY2xpZW50X2lkIjogIjEwOTc0ODMyMjk0NDg3MTY0NjAzOSIsCiAgImF1dGhfdXJpIjogImh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbS9vL29hdXRoMi9hdXRoIiwKICAidG9rZW5fdXJpIjogImh0dHBzOi8vb2F1dGgyLmdvb2dsZWFwaXMuY29tL3Rva2VuIiwKICAiYXV0aF9wcm92aWRlcl94NTA5X2NlcnRfdXJsIjogImh0dHBzOi8vd3d3Lmdvb2dsZWFwaXMuY29tL29hdXRoMi92MS9jZXJ0cyIsCiAgImNsaWVudF94NTA5X2NlcnRfdXJsIjogImh0dHBzOi8vd3d3Lmdvb2dsZWFwaXMuY29tL3JvYm90L3YxL21ldGFkYXRhL3g1MDkvc2VjdXJpY2FtLTM1MTkwNiU0MGFwcHNwb3QuZ3NlcnZpY2VhY2NvdW50LmNvbSIKfQ=="
    # CREDENTIALS = base64.b64decode(CREDENTIALS).decode('UTF-8')
    CREDENTIALS = 's-a.json'
    MODEL_PATH = 'gs://securicam-351906.appspot.com/modelv2.h5'
    FS = gcsfs.GCSFileSystem(project=PROJECT_NAME, token=CREDENTIALS)
    with FS.open(MODEL_PATH, 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        model = load_model(model_gcs, compile=False)
    print(" * Model loaded!")
    return model


def load_image(url): 
    req = request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1) # 'Load it as it is'
    # cv2.imwrite("image_loaded.jpeg", img)
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
            'confidence': img_boxes[i].confidence.item(),
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

dir_path = os.path.dirname(os.path.realpath(__file__))

app = flask.Flask(__name__)

# get_model()
load_model_from_bucket()

@app.route("/", methods=['GET'])
def base():
       response = {"success" : True, "data" : "Wellcome to securicam ai server v1", "message" : cwd, "example_result" : [{'label': 'car', 'confidence': 0.52199095, 'xmin': 2, 'ymin': 5, 'xmax': 414, 'ymax': 361}]}
       resp = flask.make_response(response, 200)
       resp.headers['Content-Type'] = 'application/json'
       return resp


@app.route("/predict/", methods=['POST'])
def predict():
       
       args = flask.request.get_json()
       image = base64.b64decode(args['image'])
       # image_url = args['image']
       # image = load_image(image_url)
       # prediction = predict_image(image)
       # filestr = flask.request.files['image'].read()
       # #convert string data to numpy array
       npimg = np.frombuffer(image, "uint8")
       prediction, result_image = predict_image(npimg)
       # response['response'] = prediction
       # return flask.jsonify(response)
       response = {"success" : True, "data" : prediction}
       print(" * Prediction result {}".format(response))
       resp = flask.make_response(response, 200)
       resp.headers['Content-Type'] = 'application/json'
       return resp
       # return flask.make_response(flask.jsonify(response), 200)
if __name__ == '__main__' :
       app.run(debug=True)
    #    app.run(debug=True, port=int(os.environ.get("PORT", 8080)))
 