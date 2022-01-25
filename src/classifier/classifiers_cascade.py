import src.classifier.vjbasic_classifier as vjbc
import src.classifier.weak_classifiers_learning as wcl
import numpy as np


class ClassifiersCascade:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.features_per_layer = [1, 10, 25, 25, 50]

        for i in range(0, n_layers - 5):
            self.features_per_layer.append(50 + i)

        self.classifiers = []

    def fit(self, X, y):
        m = wcl.load_images_from_folder(wcl.faces_path).shape[0]
        l = wcl.load_images_from_folder(wcl.non_faces_path).shape[0]
        for n in self.features_per_layer:
            classifier = vjbc.VJBasicClassifier(n)
            classifier.fit(X, y, m, l)
            self.classifiers.append(classifier)

    def add_classifier(self, classifier):
        self.classifiers.append(classifier)

    def apply(self, X):
        n = X.shape[0]
        leaves = np.ones(n)
        for classifier in self.classifiers:
            leaves = leaves * classifier.apply(X)
            if np.equal(leaves, np.zeros(n)).all():
                return leaves

        return leaves
