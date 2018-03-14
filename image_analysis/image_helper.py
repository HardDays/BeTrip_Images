from PIL import Image


def open_img(path_to_img) -> Image:
    """
    Opens image of appropriate type
    :param path_to_img: path to img
    :return: Image
    """
    f = open(path_to_img, 'r+b')
    return Image.open(f)


def open_img_for_bn_classif(path_to_img, target_size=tuple((299,299)), grayscale=False):
    img = open_img(path_to_img)
    return prepare_bn_classif(img)


def prepare_bn_classif(img: Image, target_size=tuple((299,299)), grayscale=False):
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img
