import numpy as np
import math
import src.classifier.weak_classifiers_learning as wcl
import sys
import numpy.ma as ma
from timeit import default_timer as timer


class VJBasicClassifier:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.weak_learners = []
        self.wl_feature_indexes = []
        self.alphas = []

    # TODO: add indexes of already used features to check if it improves performance?
    def fit(self, X, y, m, l, classif_apply_results, classifiers_list, features_array):
        n = X.shape[0]
        w = np.ones(n)
        self.weak_learners = []
        self.alphas = []
        n_classifiers = classif_apply_results.shape[0]
        
        start_time = timer()
        for t in range(0, self.n_estimators):
            w_sum = np.sum(w, 0)
            for i in range(0, m + l):
                w[i] = w[i] / w_sum

            err_array = np.zeros(n_classifiers)
            for j in range(0, n_classifiers):
                for i in range(0, n):
                    e = abs(classif_apply_results[j, i] - 1 - y[i])
                    err_array[j] += w[i] * e

            err_array = self.mask_errors_array(err_array)
            ht_f_index = np.argmin(err_array[~err_array.mask])
            ht = classifiers_list[ht_f_index]
            et = max(err_array[ht_f_index], sys.float_info.min)
            bt = et / (1 - et)
            for i in range(0, m + l):
                e = abs(ht.apply([[features_array[i, ht_f_index]]])[0] - 1 - y[i])
                w[i] = w[i] * pow(bt, 1 - e)

            self.weak_learners.append(ht)
            self.wl_feature_indexes.append(ht_f_index)
            self.alphas.append(math.log(1 / bt))
        end_time = timer()
        print(f'One feature classifier AB: {end_time - start_time} s')

    def apply(self, X):
        if len(self.alphas) == 0 or len(self.weak_learners) == 0:
            raise RuntimeError("can't apply classifier before fitting it.")

        X = wcl.get_train_features_dataset(X)
        n = X.shape[0]
        leaves = np.zeros(n)
        for i in range(0, n):
            classif_sum = 0
            alphas_sum = 0
            for t in range(0, self.n_estimators):
                f_index = self.wl_feature_indexes[t]
                classif_sum += self.alphas[t] * (self.weak_learners[t].apply([[X[i, f_index]]])[0] - 1)
                alphas_sum += self.alphas[t]

            if classif_sum >= 0.5 * alphas_sum:
                leaves[i] = 1
            else:
                leaves[i] = 0

        return leaves

    def mask_errors_array(self, array):
        mask = np.zeros(array.shape[0])
        for index in self.wl_feature_indexes:
            mask[index] = 1
        return ma.array(array, mask=mask, fill_value=sys.float_info.max)





