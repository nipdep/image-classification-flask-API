from flask import Flask, request, Response
import jsonpickle
from numpy import fromstring
from cv2 import imdecode,IMREAD_COLOR

from app.model import *

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/', methods=['POST'])
def index():
    r = request
    # convert string of image data to uint8
    nparr = fromstring(r.data, np.uint8)
    # decode image
    img = imdecode(nparr, IMREAD_COLOR)

    # do some fancy processing here....
    model = Model()
    lable = model.predict(img)

    # build a response dict to send back to client
    response = {'message': 'image received. lable={:s}'.format(lable)
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
#app.run(host="0.0.0.0", port=5000)

