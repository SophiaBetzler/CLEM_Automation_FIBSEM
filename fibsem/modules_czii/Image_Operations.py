import numpy as np
import cv2


class ImageOperations:
    #def __init__(self):

    def measure_image_shift(self, image0, image1):
        img1 = np.float32(image0.data)
        img2 = np.float32(image1.data)
        shift, response = cv2.phaseCorrelate(img1, img2)
        return shift

