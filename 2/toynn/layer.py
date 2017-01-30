import numpy as np


class Layer(object):
    def __init__(self, size):
        self.size = size
        self.prev = None
        self.next = None


class InputLayer(Layer):
    def __init__(self, size):
        super().__init__(size)
        self.y = None


class DenseLayer(Layer):
    def __init__(self, previous_layer, size):
        super().__init__(size)
        self.prev = previous_layer
        self.prev.next = self
        self.w = np.empty((self.prev.size, size))
        self.a = None
        self.y = None

    def f(self):
        """Evaluate f.

        Assume self.a is given.
        """
        raise NotImplementedError

    def f_prime(self):
        """Evaluate f prime.

        Assume self.a and self.y is given.
        """
        raise NotImplementedError

    def output_delta(self, t):
        """Delta when this is the output layer.

        Assume self.y is given. Assume a cross entropy loss function.
        """
        raise NotImplementedError


class LogisticLayer(DenseLayer):
    def f(self):
        return 1 / (1 + np.exp(-self.a))

    def f_prime(self):
        return self.y * (1 - self.y)


class SigmoidLayer(DenseLayer):
    def f(self):
        return 1.7159 * np.tanh(2 / 3 * self.a)

    def f_prime(self):
        a = 1.7159
        b = 2 / 3
        return a * b - a / b * self.y ** 2


class SoftmaxLayer(DenseLayer):
    def f(self):
        exp = np.exp(self.a)
        return exp / exp.sum(axis=1)[:, np.newaxis]

    def output_delta(self, t):
        return self.y - t
