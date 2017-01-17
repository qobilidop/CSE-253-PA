import numpy as np


def log(x):
    # Avoid infinity
    small = 1e-12
    return np.log(np.maximum(x, small))


class NeuralNetwork(object):
    def __init__(self, inputs, targets):
        if targets[0].size > 1:
            self.weights = np.zeros((inputs[0].size, targets[0].size))  # D * K
        else:
            self.weights = np.zeros(inputs[0].size)  # D
        self.inputs = inputs  # N * D
        self.targets = targets  # N * K

    def use_labeled_images(self, data_set):
        self.inputs = data_set.images
        self.targets = data_set.labels

    def activation_function(self, z):
        raise NotImplementedError

    def gradient(self):
        x = self.inputs  # N * D
        t = self.targets  # N * K
        y = self.outputs()  # N * K
        return -np.tensordot(x, (t - y), axes=[0, 0]) / x.shape[0]  # D * K

    def outputs(self):
        f = self.activation_function
        x = self.inputs  # N * D
        w = self.weights  # D * K
        return f(np.tensordot(x, w, axes=[1, 0]))  # N * K

    def update(self, rate, regularization=None, lam=0):
        gradient = self.gradient()
        if regularization == 'L1':
            gradient += lam * np.sign(self.weights)
        elif regularization == 'L2':
            gradient += lam * 2 * self.weights
        self.weights = self.weights - rate * gradient


class Logistic(NeuralNetwork):
    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))

    def loss_function(self):
        t = self.targets  # N
        y = self.outputs()  # N
        return -(t * log(y) + (1 - t) * log(1 - y)).mean()

    def percent_correct(self):
        t = self.targets  # N
        y = self.outputs()  # N
        correct = ((t == 1) & (y > 0.5)) | ((t == 0) & (y < 0.5))
        return correct.sum() / correct.size


class Softmax(NeuralNetwork):
    def activation_function(self, z):
        exp = np.exp(z)
        return exp / np.expand_dims(exp.sum(axis=1), axis=1)

    def loss_function(self):
        t = self.targets  # N * K
        y = self.outputs()  # N * K
        return -(t * log(y)).sum(axis=1).mean()

    def percent_correct(self):
        t = self.targets  # N * K
        y = self.outputs()  # N * K
        correct = (t.argmax(axis=1) == y.argmax(axis=1))
        return correct.sum() / correct.size
