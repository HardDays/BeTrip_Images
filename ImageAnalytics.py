from PIL import Image
from image_analysis.image_helper import prepare_bn_classif
from image_analysis.beauty_detection.BeautyClassificator import BeautyClassificator
from image_analysis.face_detection.FaceDetector import FaceDetector


class ImageAnalytics:
    def __init__(self):
        self.fd = FaceDetector()
        self.bc = BeautyClassificator()
    
    def get_score(self, img: Image):
        score = 0
        score += self._face_score(img)
        img = prepare_bn_classif(img)
        score += self._beauty_score(img)
        return score

    def _face_score(self, img):
        face = self.fd.is_with_faces(img)
        return int(not face)

    def _beauty_score(self, img):
        beautiful, prob = self.bc.is_beautiful(img)
        return int(beautiful) * prob


from image_analysis.image_helper import open_img
ia = ImageAnalytics()
im = open_img("/Users/maria/BeTrip/instagram/1/1/1.jpg")
score = ia.get_score(im)
print(score)