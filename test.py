"""

Keras Cifar-10 Classification

"""
import time
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")
# Import Tensorflow with multiprocessing for use 16 cores on plon.io
from keras.datasets import cifar10

import tensorflow as tf
import multiprocessing as mp
from keras.models import model_from_json
(x_train, y_train), (x_test, y_test) = cifar10.load_data() # x_train - training data(images), y_train - labels(digits)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
load_model("model.h5")
print("Loaded model from disk")

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('c.jpeg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
print (result)
