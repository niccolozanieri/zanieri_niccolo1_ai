import cv2
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier

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


def get_train_dataset(faces, non_faces):
    faces_images = load_images_from_folder(faces_path)
    non_faces_images = load_images_from_folder(non_faces_path)

    samples = np.concatenate((faces_images, non_faces_images))
    labels = np.concatenate((np.ones(faces_images.shape[0]), np.zeros(non_faces_images.shape[0])))

    return samples, labels



