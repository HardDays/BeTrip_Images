import face_recognition
import numpy as np
import imutils
from PIL import Image
from skimage import exposure
from resizeimage import resizeimage


class FaceDetector:
    # Image representation:
    #
    #  x1,y1 ------
    #  |          |
    #  |          |
    #  |          |
    #  --------x2,y2

    #  top ------
    #  left       |
    #  | x        |
    #  |          |
    #  |      bottom
    #    ------right
    #      y

    def is_with_faces(self, img: Image):
        """
        Indicates whether there were faces detected on the image
        :param img: Image object
        :return: boolean indicator: True if there are faces on the image
        """
        try:
            if self._is_selfie(*self._find_faces(img)):
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return True
    
    def _find_faces(self, image: Image):
        """
        Looks for faces on the supplied image
        :param image: Image
        :return: image rotated to the right angle, faces coordinates
        """
        try:
            im = self._compress(image)
        except Exception as e:
            im = np.array(image)
        # Magic with exposure
        try:
            p2, p98 = np.percentile(im, (2, 98))
            im = exposure.rescale_intensity(im, in_range=(p2, p98))
        except Exception as e:
            pass

        # Let's try simple angles first
        r_im = im
        r_angles = [0, 90, -90, 180]
        right_angle = 0
        right_faces = []
        max_square = 0

        for a in r_angles:
            max_square, right_angle, right_faces = self._rotate_and_get_faces(a, im, max_square, right_angle,
                                                                              right_faces)
        # If there were nothing found
        # we can try with other angles
        if len(right_faces) == 0:
            r_angles1 = r_angles
            r_angles = [i for i in range(0, 360, 10) if i not in r_angles1]
            for a in r_angles:
                max_square, right_angle, right_faces = self._rotate_and_get_faces(a, im, max_square, right_angle,
                                                                                  right_faces)

        if right_faces is not None:
            r_im = imutils.rotate_bound(im, right_angle)
        return r_im, right_faces

    def _rotate_and_get_faces(self, a, im, max_square, right_angle, right_faces):
        image = imutils.rotate_bound(im, a)
        # Find all the faces in the image
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) > 0:
            # Are those faces taking more space?
            sq = self._count_square(face_locations)
            if sq > max_square:
                max_square = sq
                right_angle = a
                right_faces = face_locations

        return max_square, right_angle, right_faces

    def _count_square(self, face_locations):
        sq = 0
        for face_location in face_locations:
            top, right, bottom, left = face_location
            sq += (bottom - top) * (right - left)
        return sq

    def _is_selfie(self, im, coords):
        sq_im = im.shape[0] * im.shape[1]
        sq_f = 0

        im_center = (im.shape[1] / 2, im.shape[0] / 2)

        for (top, right, bottom, left) in coords:
            face_square = abs((right - left) * (bottom - top))
            # If face square is 0.1 of image square it is surely selfie
            if face_square > 0.1 * sq_im:
                return True
            # If face square is between 0.001 and 0.1 of image square it
            # could be selfie if it is in the center
            if face_square > 0.001 * sq_im:
                if self._is_in_center(im, im_center, (top, right, bottom, left)):
                    print('Selfie!')
                    return True
            sq_f += face_square

        frq = sq_f / sq_im

        print('Square of image: %f' % sq_im)
        print('Square of faces: %f' % sq_f)
        print('Fraction of face on image %f' % frq)
        print('Fraction of selfie is %f ' % (1 / 3))

        # If summary of squares of all faces is more then 0.1 of image
        # square it is surely selfie
        if frq >= 0.1:
            print('Selfie!')
            return True

        print('Not selfie!')
        return False

    def _is_in_center(self, im, center, face_coords):
        (top, right, bottom, left) = face_coords

        e = self._get_derivation(im.shape, *face_coords)
        x1 = center[1] - e[1]
        x2 = center[1] + e[1]
        y1 = center[0] - e[0]
        y2 = center[0] + e[0]

        if (x1 < top < x2 or x1 < bottom < x2) and (y1 < left < y2 or y1 < right < y2):
            return True
        return False

    def _get_derivation(self, im_shape, top, right, bottom, left):
        der = (abs((bottom - top) * 2.2), abs((right - left) * 2.2))
        k = im_shape[1] / im_shape[0]
        if k < 1:
            k = 1 / k
            return der[0], der[1] * k
        return der[0] * k, der[1]

    def _compress(self, image):
        img = np.array(image)
        new_shape = self._get_new_shape(img.shape)

        if img.shape[0] == image.width:
            cover = resizeimage.resize_cover(image, [new_shape[1], new_shape[0]])
        else:
                    cover = resizeimage.resize_cover(image, [new_shape[0], new_shape[1]])
        return np.array(cover)

    def _get_new_shape(self, old_shape):
        h = 1000
        if old_shape[0] <= h:
            return (old_shape[1], old_shape[0])
        return old_shape[1] / old_shape[0] * h, 1000

