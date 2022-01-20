from classifier import weak_classifiers_learning as wcl
from classifier import vjbasic_classifier as vjbc
from timeit import default_timer as timer


start_test = timer()
test_faces_path = "/home/niccolo/Insync/niccolo.zanieri.13@gmail.com/Google Drive/School/" \
             "University/Terzo_Anno/Intelligenza Artificiale/Esame/data/test_data/faces"

test_non_faces_path = "/home/niccolo/Insync/niccolo.zanieri.13@gmail.com/Google Drive/School/" \
             "University/Terzo_Anno/Intelligenza Artificiale/Esame/data/test_data/non_faces"

(samples, labels) = wcl.get_train_images_dataset(wcl.faces_path, wcl.non_faces_path)
classifier = vjbc.VJBasicClassifier(2)
m = wcl.load_images_from_folder(wcl.faces_path).shape[0]
l = wcl.load_images_from_folder(wcl.non_faces_path).shape[0]
classifier.fit(samples, labels, m, l)
(test_samples, test_labels) = wcl.get_train_images_dataset(test_faces_path, test_non_faces_path)
pred = classifier.apply(test_samples)

end_test = timer()

print(f'Train + classification: {end_test - start_test} s')
miscl = 0
for i in range(0, test_samples.shape[0]):
    if test_labels[i] != pred[i]:
        miscl += 1

print("Misclassified: " + str(miscl))
print("Classified correctly: " + str(test_samples.shape[0] - miscl))
print(4)





