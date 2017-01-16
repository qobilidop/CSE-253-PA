import numpy as np


class NeuralNetwork(object):
    def __init__(self, dim):
        self.weights = np.zeros(dim)
        self.inputs = None
        self.targets = None

    def use_labeled_images(self, data_set):
        self.inputs = data_set.images
        self.targets = data_set.labels

    def actication_function(self):
        raise NotImplementedError

    def gradient(self):
        t = self.targets
        y = self.outputs()
        x = self.inputs
        return -np.tensordot((t - y), x, axes=[0, 0])

    def outputs(self):
        f = self.activation_function
        w = self.weights
        x = self.inputs
        return f(np.tensordot(w, x, axes=[0, 1]))

    def update(self, rate, lam=0, regularization=None):
        gradient = self.gradient()
        if regularization:
            gradient += lam * regularization
        self.weights = self.weights - rate * gradient


class Logistic(NeuralNetwork):
    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))

    def loss_function(self):
        t = self.targets
        y = self.outputs()
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
