from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:5000'
# addr = 'http://pr-img-class.herokuapp.com'
test_url = addr + '/'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
dim = (224,224)

img = cv2.imread('../data/img.jpg')
img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers)
# decode response
print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}
