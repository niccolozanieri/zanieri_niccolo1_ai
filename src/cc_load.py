from classifier import weak_classifiers_learning as wcl
from timeit import default_timer as timer
import pickle

test_faces_path = "../data/test_data/faces"

test_non_faces_path = "../data/test_data/non_faces"


with open('pickled_classifiers/classifiers_cascade.pickle', 'rb') as input_file:
    classifier = pickle.load(input_file)

(test_samples, test_labels) = wcl.get_images_dataset(test_faces_path, test_non_faces_path)

start_test = timer()
pred = classifier.apply(test_samples)
end_test = timer()

miscl = 0
false_positives = 0
for i in range(0, test_samples.shape[0]):
    if test_labels[i] != pred[i]:
        miscl += 1
        if pred[i] == 1:
            false_positives += 1

print(f'Average classification time per sub-window: {(end_test - start_test) / test_samples.shape[0]} s')
print("Misclassified: " + str(miscl))
print("False positives: " + str(false_positives))
print("Classified correctly: " + str(test_samples.shape[0] - miscl))


