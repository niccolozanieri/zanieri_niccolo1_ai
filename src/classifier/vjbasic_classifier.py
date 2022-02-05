import numpy as np
import math
import src.haar_features.haar_like_features as hlf
import sys


class VJBasicClassifier:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.weak_learners = []
        self.wl_features = []
        self.alphas = []

    def fit(self, X, y, m, l, classif_apply_results, classifiers_list, features_array):
        n = X.shape[0]
        w = np.ones(n)
        self.weak_learners = []
        self.alphas = []

        y_mat = np.zeros((classif_apply_results.shape[0], classif_apply_results.shape[1]))
        for j in range(0, y_mat.shape[1]):
            y_mat[:, j] = y[j]

        for i in range(0, m):
            w[i] = w[i] / (2 * m)
        for j in range(0, l):
            w[m + j] = w[m + j] / (2 * l)

        for t in range(0, self.n_estimators):
            w_sum = np.sum(w, 0)
            for i in range(0, m + l):
                w[i] = w[i] / w_sum

            e = np.abs(classif_apply_results - y_mat)
            err_array = np.matmul(e, w)

            ht_f_index = np.argmin(err_array)
            ht = classifiers_list[ht_f_index]
            et = max(err_array[ht_f_index], sys.float_info.min)
            bt = et / (1 - et)
            for i in range(0, m + l):
                e = abs(ht.predict([[features_array[i, ht_f_index].value]])[0] - y[i])
                w[i] = w[i] * pow(bt, 1 - e)

            self.weak_learners.append(ht)
            self.wl_features.append(features_array[0, ht_f_index])
            self.alphas.append(math.log(1 / bt))

    def apply(self, X):
        if len(self.alphas) == 0 or len(self.weak_learners) == 0:
            raise RuntimeError("can't apply classifier before fitting it.")

        n = X.shape[0]
        leaves = np.zeros(n)
        for i in range(0, n):
            classif_sum = 0
            alphas_sum = 0
            for t in range(0, self.n_estimators):
                ii = hlf.integral_image(X[i])
                feature = self.wl_features[t]
                f_value = hlf.get_rectangular_feature(ii, hf=feature)
                classif_sum += self.alphas[t] * (self.weak_learners[t].predict([[f_value]])[0])
                alphas_sum += self.alphas[t]

            if classif_sum >= 0.5 * alphas_sum:
                leaves[i] = 1
            else:
                leaves[i] = 0

        return leaves





