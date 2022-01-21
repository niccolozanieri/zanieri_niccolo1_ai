import cv2
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.haar_features import haar_like_features as hlf
from sklearn.tree import export_text

faces_path = "/home/niccolo/Insync/niccolo.zanieri.13@gmail.com/Google Drive/School/" \
             "University/Terzo_Anno/Intelligenza Artificiale/Esame/data/train_data/faces"

non_faces_path = "/home/niccolo/Insync/niccolo.zanieri.13@gmail.com/Google Drive/School/" \
                 "University/Terzo_Anno/Intelligenza Artificiale/Esame/data/train_data/non_faces"


# https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return np.array(images)


def get_train_images_dataset(faces, non_faces):
    faces_images = load_images_from_folder(faces)
    non_faces_images = load_images_from_folder(non_faces)

    samples = np.concatenate((faces_images, non_faces_images))
    labels = np.concatenate((np.ones(faces_images.shape[0]), np.zeros(non_faces_images.shape[0])))

    return samples, labels


def get_image_variance(src_image):
    h = src_image.shape[0]
    w = src_image.shape[1]
    src_image = np.array(src_image, dtype=np.uint)
    src_image_square = src_image ** 2
    ii = hlf.integral_image(src_image)
    ii_square = hlf.integral_image(src_image_square)
    mean = ii[h - 1, w - 1] / (h * w)
    mean_square = ii_square[h - 1, w - 1] / (h * w)
    variance = mean_square - mean ** 2

    return variance


def variance_normalize(images, features):
    for i, img in enumerate(images):
        features[i] = features[i] / get_image_variance(img)


def get_train_features_dataset(images):
    features = []

    for img in images:
        features.append(hlf.get_rectangular_features_24(img))

    return np.array(features)


def train_features_classifiers(X, y):
    features = get_train_features_dataset(X)
    classifiers = []
    for i in range(0, features.shape[1]):
        samples = np.zeros(features.shape[0])
        for j in range(0, features.shape[0]):
            samples[j] = features[j, i]
        samples = samples.reshape(-1, 1)
        dt = DecisionTreeClassifier(max_depth=1)
        dt = dt.fit(samples, y)
        classifiers.append(dt)

    return classifiers, features

