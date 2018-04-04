from .ImageAnalytics import ImageAnalytics
from .image_analysis.image_helper import open_img


ia = ImageAnalytics()
im = open_img("../../instagram/1/1/1.jpg")
score = ia.get_score(im)
print(score)