from collections import namedtuple

from mnist import MNIST
import numpy as np


def read_data_sets(directory, one_hot=False):
    FullSets = namedtuple('FullSets', ['train', 'test', 'validation'])

    mndata = MNIST(directory)

    train_raw = mndata.load_training()
    boundary = 55000
    train = DataSet(images=train_raw[0][:boundary],
                    labels=train_raw[1][:boundary])
    validation = DataSet(images=train_raw[0][boundary:],
                         labels=train_raw[1][boundary:])

    test_raw = mndata.load_testing()
    test = DataSet(images=test_raw[0], labels=test_raw[1])

    if one_hot:
        for data_set in [train, test, validation]:
            size = len(data_set.labels)
            one_hot_labels = np.zeros((size, 10))
            one_hot_labels[range(size), data_set.labels] = 1
            data_set.labels = one_hot_labels

    return FullSets(train=train, test=test, validation=validation)


class DataSet(object):
    def __init__(self, images, labels):
        self.images = np.array(images)
        self.labels = np.array(labels)

    @property
    def dim(self):
        return self.images.shape[1]

    @property
    def size(self):
        return self.images.shape[0]
