import unittest
import numpy as np

from src.haar_features.haar_like_features import integral_image


class IntegralImageTest(unittest.TestCase):
    def test_ii(self):
        # example taken from https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/
        test_image = np.array([[5, 2, 5, 2], [3, 6, 3, 6], [5, 2, 5, 2], [3, 6, 3, 6]])
        expected_ii = np.array([[5, 7, 12, 14], [8, 16, 24, 32], [13, 23, 36, 46], [16, 32, 48, 64]])
        ii = integral_image(test_image)

        self.assertEqual(True, np.array_equal(ii, expected_ii))

    def test_image_sum_ii(self):
        test_image = np.array([[5, 2, 5, 2], [3, 6, 3, 6], [5, 2, 5, 2], [3, 6, 3, 6]])
        pixels_sum = 0
        h = test_image.shape[0]
        w = test_image.shape[1]
        for i in range(0, h):
            for j in range(0, w):
                pixels_sum += test_image[i, j]

        ii = integral_image(test_image)

        self.assertEqual(pixels_sum, ii[h - 1, w - 1])


