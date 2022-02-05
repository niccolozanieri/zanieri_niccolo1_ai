import src.classifier.vjbasic_classifier as vjbc
import src.classifier.weak_classifiers_learning as wcl
import numpy as np


def get_classif_apply_results(X, y):
    n = X.shape[0]
    (classifiers_list, features_array) = wcl.train_features_classifiers(X, y)

    classif_apply_results = np.zeros((len(classifiers_list), n))
    for j in range(0, len(classifiers_list)):
        for i in range(0, n):
            classif_apply_results[j, i] = classifiers_list[j].predict([[features_array[i, j].value]])[0]

    return classif_apply_results, classifiers_list, features_array


class ClassifiersCascade:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        x = [1, 10, 25, 25, 50]
        if n_layers < 5:
            self.features_per_layer = x[0: n_layers]
        else:
            self.features_per_layer = x

        for i in range(0, n_layers - 5):
            self.features_per_layer.append(50 + i)

        self.classifiers = []

    def fit(self, X, y):
        (classif_apply_results, classifiers_list, features_array) = get_classif_apply_results(X, y)

        for n in self.features_per_layer:
            m = (y == 1).sum()
            l = (y == 0).sum()
            classifier = vjbc.VJBasicClassifier(n)
            classifier.fit(X, y, m, l, classif_apply_results, classifiers_list, features_array)
            self.classifiers.append(classifier)
            r = []
            for i in range(0, X.shape[0]):
                if (self.sub_window_apply(np.array([X[i]])) == 1 and y[i] == 0) or (y[i] == 1):
                    r.append(i)
            X = X[r, :, :]
            y = y[r]
            classif_apply_results = classif_apply_results[:, r]
            features_array = features_array[r, :]

    def apply(self, X):
        n = X.shape[0]
        leaves = np.ones(n)
        for i in range(0, n):
            leaves[i] = self.sub_window_apply(np.array([X[i]]))
        return leaves

    def sub_window_apply(self, sub_window):
        for classifier in self.classifiers:
            if classifier.apply(sub_window)[0] == 0:
                return 0
        return 1
