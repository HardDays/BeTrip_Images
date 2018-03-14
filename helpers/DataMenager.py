import os
# from .Paths import model_weights, model_name
from helpers import Paths
from keras.models import model_from_json


class DataMenager:
    bn_model = None
    
    def __init__(self):
        self._basepath = os.path.dirname(__file__)
        if not self.bn_model:
            self.bn_model = self._import_bn_model()
    
    def _get_path_from_root(self, path):
        return os.path.abspath(os.path.join(self._basepath, "..", path))
    
    def _import_bn_model(self):
        mn = self._get_path_from_root(Paths.model_name)
        mw = self._get_path_from_root(Paths.model_weights)
        
        json_file = open(mn, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(mw)
        return model
