from PIL import Image
from image_analysis.image_helper import prepare_bn_classif
from image_analysis.beauty_detection.BeautyClassificator import BeautyClassificator
from image_analysis.face_detection.FaceDetector import FaceDetector


class ImageAnalytics:
    def __init__(self):
        self.fd = FaceDetector()
        self.bc = BeautyClassificator()
    
    def get_score(self, img: Image):
        """
        Returns score for image depending on it's beauty and containing of faces
        :param img: Image (PIL)
        :return: score in range [0,2]
        """
        return self._face_score(img) + self._beauty_score(prepare_bn_classif(img))

    def _face_score(self, img):
        """
        Gives image score 0 if it contains faces and 1 otherwise
        :param img: Image
        :return: 0 or 1
        """
        face = self.fd.is_with_faces(img)
        return int(not face)

    def _beauty_score(self, img):
        """
        Gives image a score according to its beauty. Higher score corresponds to more beautiful images.
        :param img: Image
        :return: score in range [0,1]
        """
        beautiful, prob = self.bc.is_beautiful(img)
        return int(beautiful) * prob


# from image_analysis.image_helper import open_img
# ia = ImageAnalytics()
# im = open_img("/Users/maria/BeTrip/instagram/1/1/1.jpg")
# score = ia.get_score(im)
# print(score)