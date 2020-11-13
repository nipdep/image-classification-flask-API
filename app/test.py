
import cv2



img = cv2.imread('../data/img.jpg', cv2.IMREAD_UNCHANGED)
print(img.shape[0])
# img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
