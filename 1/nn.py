import numpy as np


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

    def activation_function(self):
        raise NotImplementedError

    def gradient(self):
        x = self.inputs  # N * D
        t = self.targets  # N * K
        y = self.outputs()  # N * K
        N = self.inputs.shape[0]
        return -np.tensordot(x, (t - y), axes=[0, 0]) / N  # D * K

    def outputs(self):
        f = self.activation_function
        x = self.inputs  # N * D
        w = self.weights  # D * K
        return f(np.tensordot(x, w, axes=[1, 0]))  # N * K

    def update(self, rate, lam=0, regularization=None):
        gradient = self.gradient()
        if regularization:
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
        # We want to make sure log(y) won't go crazy.
        threshold = 1e-12
        y[y < threshold] = threshold
        y[y > 1 - threshold] = 1 - threshold
        return -(t * np.log(y) + (1 - t) * np.log(1 - y)).mean()

    def percent_correct(self):
        y = self.outputs()
        t = self.targets
        correct = ((y > 0.5) & (t == 1)) | ((y < 0.5) & (t == 0))
        return correct.sum() / correct.size


class Softmax(NeuralNetwork):
    def activation_function(self, z):
        exp = np.exp(z)
        return exp / np.expand_dims(exp.sum(axis=1), axis=1)

    def loss_function(self):
        t = self.targets  # N * K
        y = self.outputs()  # N * K
        # We want to make sure log(y) won't go crazy.
        threshold = 1e-12
        y[y < threshold] = threshold
        return -(t * np.log(y)).sum(axis=1).mean()

    def percent_correct(self):
        t = self.targets  # N * K
        y = self.outputs()  # N * K
        correct = (t.argmax(axis=1) == y.argmax(axis=1))
        return correct.sum() / correct.size
