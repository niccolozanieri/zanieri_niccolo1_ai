import cv2
import numpy as np


def integral_image(src_image):
    h = src_image.shape[0]
    w = src_image.shape[1]

    ii = np.zeros((h, w))
    s = np.zeros((h, w))

    for y in range(0, h):
        for x in range(0, w):
            if y == 0:
                s[y, x] = src_image[y, x]
            else:
                s[y, x] = s[y - 1, x] + src_image[y, x]

            if x == 0:
                ii[y, x] = s[y, x]
            else:
                ii[y, x] = ii[y, x - 1] + s[y, x]

    return ii



