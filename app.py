
import flask
from keras.models import load_model

from utils import get_model

app = flask.Flask(__name__)

print('Load model...')
get_model()


@app.route("/predict/", methods=['GET', 'POST'])
def predict():
       response = {"success" : True}
       
       return response

if __name__ == '__main__' :
       app.run(debug=True, port=8000)
