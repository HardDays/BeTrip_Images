import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils.np_utils import probas_to_classes
from keras import backend as K
import os

from helpers.DataMenager import DataMenager


class BeautyClassificator(object):
    img_width, img_height = 299, 299
    batch_size = 32
    
    def __init__(self):
        self.model = DataMenager().bn_model
        self.image_data_generator = ImageDataGenerator(rescale=1. / 255)
    
    def is_beautiful(self, img: Image):
        x = img_to_array(img, dim_ordering=K.image_dim_ordering())
        x = self.image_data_generator.random_transform(x)
        x = self.image_data_generator.standardize(x)
        y_probabilities = self.model.predict(x=np.array([x]), batch_size=self.batch_size)
        y_classes = probas_to_classes(y_probabilities)
        return y_classes[0], y_probabilities.max()