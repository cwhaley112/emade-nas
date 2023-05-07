'''
This script downloads the CIFAR10 dataset. 

The train set is split to create a validation set. This is 
given to EMADE as the test set so that our NNs are unbiased when
we train them on the actual test data.
'''

import numpy as np
import random
import sklearn.model_selection
import tensorflow_datasets as tfds

random.seed(111)
np.random.seed(111)

ds = tfds.load("cifar10", split="train", as_supervised=True)
images = []
labels = []
for image, label in ds.as_numpy_iterator():
    images.append(image)
    labels.append(label)
images = np.array(images)
labels = np.array(labels)

ds_test = tfds.load("cifar10", split="test", as_supervised=True)
images_test = []
labels_test = []
for image, label in ds_test.as_numpy_iterator():
    images_test.append(image)
    labels_test.append(label)
images_test = np.array(images_test)
labels_test = np.array(labels_test)

sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.15)
for i, (train_index, test_index) in enumerate(sss.split(images, labels)):
    train_set = [
        [image, np.array([truth])]
        for image, truth in zip(images[train_index], labels[train_index])
    ]
    valid_set = [
        [image, np.array([truth])]
        for image, truth in zip(images[test_index], labels[test_index])
    ]

    np.savez_compressed("./emade_train_cifar10_" + str(i), arr=train_set)
    np.savez_compressed("./emade_test_cifar10_" + str(i), arr=valid_set)

test_data = []
for i in range(0, images_test.shape[0]):
    emade_test_instance = [images_test[i][:, :, :], np.array([labels_test[i]])]
    test_data.append(emade_test_instance)

np.savez_compressed("./test_cifar10", arr=np.array(test_data))