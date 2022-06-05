import base64
import io
from urllib import request

import flask
from keras.models import load_model
from PIL import Image
from werkzeug.datastructures import ImmutableMultiDict

from utils import get_model, preprocess_image

app = flask.Flask(__name__)

model = get_model()

@app.route("/predict/", methods=['POST'])
def predict():
       response = {"success" : True}
       print("Posted file: {}".format(flask.request.files['image']))
       message = flask.request.files['image']
       file = message
       # decoded = base64.b64decode(encoded)
       # image = Image.open(io.BytesIO(decoded))
       image = Image.open(file)
       processed_image = preprocess_image(image, target_size=(416, 416))
       prediction = model.predict(processed_image).tolist()
       response['result'] = prediction
       return flask.jsonify(response)

if __name__ == '__main__' :
       app.run(debug=True, port=8000)
