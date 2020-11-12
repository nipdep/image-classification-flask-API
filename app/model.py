
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# import os

class Model:


    def __init__(self,model = "model", weights = "weights"):
        self.root = "./data/"
        self.model_path = self.root+model+".h5"
        self.weights_path = self.root+weights+".h5"
        self.clToInt_dict={0: 'Brain', 1: 'Eye', 2: 'Heart', 3: 'Kidney', 4: 'Other', 5: 'Skeleton'}


    def predict(self, x_img):
        model = load_model(self.model_path)
        model.load_weights(self.weights_path)

        # x_img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(x_img,)
        x = np.expand_dims(x, axis=0)
        result = model.predict(x)
        img_class = np.argmax(result[0])
        str_img_class = self.clToInt_dict[img_class]

        return str_img_class



# sample = Sampling()
