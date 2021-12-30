import unittest
import numpy as np

from src.main import integral_image


class IntegralImageTest(unittest.TestCase):
    def test_ii(self):
        # example taken from https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/
        test_image = np.array([[5, 2, 5, 2], [3, 6, 3, 6], [5, 2, 5, 2], [3, 6, 3, 6]])
        expected_ii = np.array([[5, 7, 12, 14], [8, 16, 24, 32], [13, 23, 36, 46], [16, 32, 48, 64]])
        ii = integral_image(test_image)

        self.assertEqual(True, np.array_equal(ii, expected_ii))

