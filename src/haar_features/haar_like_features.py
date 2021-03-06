import numpy as np
import warnings
from src.haar_features.haar_feature import HaarFeature
warnings.filterwarnings("error")


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


def get_grey_sum(ii, size_x, size_y, w, h, x, y):
    shape = (size_x, size_y)

    if shape == (1, 2):
        grey_sum_plus = ii[y + int(h / 2) - 1, x + w - 1] + ii[y - 1, x - 1]
        grey_sum_minus = ii[y - 1, x + w - 1] + ii[y + int(h / 2) - 1, x - 1]
        grey_sum = grey_sum_plus - grey_sum_minus
    elif shape == (2, 1):
        grey_sum_plus = ii[y + h - 1, x + w - 1] + ii[y - 1, x + int(w / 2) - 1]
        grey_sum_minus = (ii[y + h - 1, x + int(w / 2) - 1] + ii[y - 1, x + w - 1])
        grey_sum = grey_sum_plus - grey_sum_minus

    elif shape == (3, 1):
        grey_sum_plus = ii[y + int(2 * h / 3) - 1, x + w - 1] + ii[y + int(h / 3) - 1, x - 1]
        grey_sum_minus = ii[y + int(h / 3) - 1, x + w - 1] + ii[y + int(2 * h / 3) - 1, x - 1]
        grey_sum = grey_sum_plus - grey_sum_minus
    elif shape == (1, 3):
        grey_sum_plus = ii[y + h - 1, x + int(2 * w / 3) - 1] + ii[y - 1, x + int(w / 3) - 1]
        grey_sum_minus = ii[y - 1, x + int(2 * w / 3) - 1] + ii[y + h - 1, x + int(w / 3) - 1]
        grey_sum = grey_sum_plus - grey_sum_minus
    else:
        grey_sum_1_plus = ii[y + int(h / 2) - 1, x + w - 1] + ii[y - 1, x + int(w / 2) - 1]
        grey_sum_1_minus = ii[y - 1, x + w - 1] + ii[y + int(h / 2) - 1, x + int(w / 2) - 1]
        grey_sum_2_plus = ii[y + h - 1, x + int(w / 2) - 1] + ii[y + int(h / 2) - 1, x - 1]
        grey_sum_2_minus = ii[y + int(h / 2) - 1, x + int(w / 2) - 1] + ii[y + h - 1, x - 1]
        grey_sum = grey_sum_1_plus + grey_sum_2_plus - grey_sum_1_minus - grey_sum_2_minus

    return grey_sum


def get_white_sum(ii, size_x, size_y, w, h, x, y):
    shape = (size_x, size_y)

    if shape == (1, 2):
        white_sum_plus = ii[y + h - 1, x + w - 1] + ii[y + int(h / 2) - 1, x - 1]
        white_sum_minus = ii[y + int(h / 2) - 1, x + w - 1] + ii[y + h - 1, x - 1]
        white_sum = white_sum_plus - white_sum_minus
    elif shape == (2, 1):
        white_sum_plus = ii[y + h - 1, x + int(w / 2) - 1] + ii[y - 1, x - 1]
        white_sum_minus = (ii[y - 1, x + int(w / 2) - 1] + ii[y + h - 1, x - 1])
        white_sum = white_sum_plus - white_sum_minus
    elif shape == (3, 1):
        white_sum_1_plus = ii[y + int(h / 3) - 1, x + w - 1] + ii[y - 1, x - 1]
        white_sum_1_minus = ii[y - 1, x + w - 1] + ii[y + int(h / 3) - 1, x - 1]
        white_sum_2_plus = ii[y + h - 1, x + w - 1] + ii[y + int(2 * h / 3) - 1, x - 1]
        white_sum_2_minus = ii[y + int(2 * h / 3) - 1, x + w - 1] + ii[y + h - 1, x - 1]
        white_sum = white_sum_1_plus + white_sum_2_plus - white_sum_1_minus - white_sum_2_minus
    elif shape == (1, 3):
        white_sum_1_plus = ii[y + h - 1, x + int(w / 3) - 1] + ii[y - 1, x - 1]
        white_sum_1_minus = ii[y - 1, x + int(w / 3) - 1] + ii[y + h - 1, x - 1]
        white_sum_2_plus = ii[y + h - 1, x + w - 1] + ii[y - 1, x + int(2 * w / 3) - 1]
        whit_sum_2_minus = ii[y - 1, x + w - 1] + ii[y + h - 1, x + int(2 * w / 3) - 1]
        white_sum = white_sum_1_plus + white_sum_2_plus - white_sum_1_minus - whit_sum_2_minus
    else:
        white_sum_1_plus = ii[y + int(h / 2) - 1, x + int(w / 2) - 1] + ii[y - 1, x - 1]
        white_sum_1_minus = ii[y - 1, x + int(w / 2) - 1] + ii[y + int(h / 2) - 1, x - 1]
        white_sum_2_plus = ii[y + h - 1, x + w - 1] + ii[y + int(h / 2) - 1, x + int(w / 2) - 1]
        white_sum_2_minus = ii[y + int(h / 2) - 1, x + w - 1] + ii[y + h - 1, x + int(w / 2) - 1]
        white_sum = white_sum_1_plus + white_sum_2_plus - white_sum_1_minus - white_sum_2_minus

    return white_sum


def get_rectangular_feature(ii, size_x=1, size_y=1, w=1, h=1, x=0, y=0, hf=None):
    mod_ii = np.zeros((25, 25))
    mod_ii[1:25, 1:25] = np.copy(ii)
    if hf is None:
        grey_sum = get_grey_sum(mod_ii, size_x, size_y, w, h, x + 1, y + 1)
        white_sum = get_white_sum(mod_ii, size_x, size_y, w, h, x + 1, y + 1)
    else:
        grey_sum = get_grey_sum(mod_ii, hf.size_x, hf.size_y, hf.w, hf.h, hf.x + 1, hf.y + 1)
        white_sum = get_white_sum(mod_ii, hf.size_x, hf.size_y, hf.w, hf.h, hf.x + 1, hf.y + 1)

    return grey_sum - white_sum


def get_image_variance(src_image, ii):
    h = src_image.shape[0]
    w = src_image.shape[1]
    src_image = np.array(src_image, dtype=np.uint)
    src_image_square = src_image ** 2
    ii_square = integral_image(src_image_square)
    mean = ii[h - 1, w - 1] / (h * w)
    mean_square = ii_square[h - 1, w - 1] / (h * w)
    variance = mean_square - mean ** 2

    return variance


# https://stackoverflow.com/questions/1707620/viola-jones-face-detection-claims-180k-features
def get_rectangular_features_24(src_image):
    if src_image.shape[0] != 24 or src_image.shape[1] != 24:
        raise ValueError("src_image must have 24x24 dimensions.")

    features_num = 5
    features_shape = np.array([[1, 2], [2, 1], [1, 3], [3, 1], [2, 2]])
    frame_size = 24
    features = []
    ii = integral_image(src_image)
    variance = get_image_variance(src_image, ii)
    if variance == 0:
        variance = 1

    for i in range(0, features_num):
        size_x = features_shape[i, 0]
        size_y = features_shape[i, 1]

        # each size (multiples of basic shapes)
        for w in range(size_x, frame_size + 1, size_x):
            for h in range(size_y, frame_size + 1, size_y):

                # each possible position given size
                for x in range(0, frame_size - w + 1):
                    for y in range(0, frame_size - h + 1):
                        value = get_rectangular_feature(ii, size_x, size_y, w, h, x, y) / variance
                        features.append(HaarFeature(size_x, size_y, x, y, h, w, value))

    return features
