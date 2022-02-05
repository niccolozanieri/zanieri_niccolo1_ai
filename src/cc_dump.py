from classifier import weak_classifiers_learning as wcl
from timeit import default_timer as timer
from classifier import classifiers_cascade as cc
import pickle

start_test = timer()

test_faces_path = "../data/test_data/faces"

test_non_faces_path = "../data/test_data/non_faces"

(samples, labels) = wcl.get_images_dataset(wcl.faces_path, wcl.non_faces_path)
classifier = cc.ClassifiersCascade(5)
classifier.fit(samples, labels)
end_test = timer()
print(f'Train: {end_test - start_test} s')

with open('pickled_classifiers/classifiers_cascade1.pickle', 'wb') as output_file:
    pickle.dump(classifier, output_file, protocol=pickle.HIGHEST_PROTOCOL)


(test_samples, test_labels) = wcl.get_images_dataset(test_faces_path, test_non_faces_path)
start_test = timer()
pred = classifier.apply(test_samples)
end_test = timer()
print(f'Apply: {end_test - start_test} s')

miscl = 0
false_positives = 0
for i in range(0, test_samples.shape[0]):
    if test_labels[i] != pred[i]:
        miscl += 1
        if pred[i] == 1:
            false_positives += 1

print("Misclassified: " + str(miscl))
print("False positives: " + str(false_positives))
print("Classified correctly: " + str(test_samples.shape[0] - miscl))


