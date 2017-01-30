from mnist import MNIST
import numpy as np


class FullDataSets(object):
    def __init__(self, train, test, validation):
        self.train = train
        self.test = test
        self.validation = validation


class DataSet(object):
    def __init__(self, images, labels):
        self.images = np.array(images)
        self.images = np.insert(self.images, 0, 1, axis=1)
        self.labels = np.array(labels)

    @property
    def dim(self):
        return self.images.shape[1]

    @property
    def size(self):
        return self.images.shape[0]


def read_data_sets(one_hot=False, directory='mnist'):
    mndata = MNIST(directory)

    train_raw = mndata.load_training()
    # Use the first 20000
    images = train_raw[0][:20000]
    labels = train_raw[1][:20000]
    selection = np.arange(0, len(labels), 10)
    train = DataSet(images=np.delete(images, selection, 0),
                    labels=np.delete(labels, selection, 0))
    validation = DataSet(images=np.array(images)[selection],
                         labels=np.array(labels)[selection])
    train.images = train.images / 255
    train.images = train.images - np.mean(train.images, axis=1)[:, np.newaxis] @ np.ones((1, train.dim))
    validation.images = validation.images / 255
    validation.images = validation.images - np.mean(validation.images, axis=1)[:, np.newaxis] @ np.ones((1, validation.dim))

    test_raw = mndata.load_testing()
    # Use the first 2000
    images = test_raw[0][:2000]
    labels = test_raw[1][:2000]
    test = DataSet(images=images, labels=labels)
    test.images = test.images / 255
    test.images = test.images - np.mean(test.images, axis=1)[:, np.newaxis] @ np.ones((1, test.dim))

    if one_hot:
        for data_set in [train, test, validation]:
            size = len(data_set.labels)
            one_hot_labels = np.zeros((size, 10))
            one_hot_labels[range(size), data_set.labels] = 1
            data_set.labels = one_hot_labels

    return FullDataSets(train=train, test=test, validation=validation)

