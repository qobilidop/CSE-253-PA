from mnist import MNIST
import numpy as np


class FullDataSets(object):
    def __init__(self, training, validation, test):
        self.training = training
        self.validation = validation
        self.test = test


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

    def normalize(self):
        self.images = self.images / 255
        self.images -= self.images.mean(axis=1)[:, np.newaxis]
        return self

    def add_biases(self):
        self.images = np.insert(self.images, 0, 1, axis=1)
        return self

    def convert_to_one_hot(self):
        size = len(self.labels)
        one_hot_labels = np.zeros((size, 10))
        one_hot_labels[range(size), self.labels] = 1
        self.labels = one_hot_labels
        return self

    def shuffle(self):
        p = np.random.permutation(len(self.labels))
        self.images = self.images[p]
        self.labels = self.labels[p]
        return self

    def minibatches(self, batch_size):
        batch_num = self.size % batch_size
        return [DataSet(images=i, labels=l) for i, l in
                zip(np.array_split(self.images, batch_num),
                    np.array_split(self.labels, batch_num))]


def read_data_sets(directory='data'):
    mndata = MNIST(directory)

    training_raw = mndata.load_training()
    images = training_raw[0]
    labels = training_raw[1]
    selection = np.arange(0, len(labels), 6)
    training = DataSet(images=np.delete(images, selection, 0),
                       labels=np.delete(labels, selection, 0))
    validation = DataSet(images=np.array(images)[selection],
                         labels=np.array(labels)[selection])

    test_raw = mndata.load_testing()
    test = DataSet(images=test_raw[0], labels=test_raw[1])

    for data_set in [training, test, validation]:
        # It's important that `normalize` comes before `add_biases` so as to
        # not normalize biases.
        data_set.normalize().add_biases().convert_to_one_hot()

    return FullDataSets(training=training, validation=validation, test=test)
