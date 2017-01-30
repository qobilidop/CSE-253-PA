import numpy as np

from toynn.layer import InputLayer
from toynn.layer import LogisticLayer
from toynn.layer import SigmoidLayer
from toynn.layer import SoftmaxLayer
from toynn.util import sane_log


class _BaseNetwork(object):
    def __init__(self):
        self.layers = None

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def dense_layers(self):
        return self.layers[1:]

    @property
    def hidden_layers(self):
        return self.layers[1:-1]

    @property
    def output_layer(self):
        return self.layers[-1]


class NaiveNetwork(_BaseNetwork):
    _hidden_layer_class = LogisticLayer

    def __init__(self, sizes):
        super().__init__()
        depth = len(sizes) - 1
        layers = []
        layers += [InputLayer(sizes[0])]
        for size in sizes[1:-1]:
            layers += [self._hidden_layer_class(layers[-1], size)]
        layers += [SoftmaxLayer(layers[-1], sizes[-1])]
        self.depth = depth
        self.layers = layers
        self.x = None
        self.t = None
        self.y = None

    @property
    def loss(self):
        """Cross entropy loss."""
        return -(self.t * sane_log(self.y)).sum(axis=1).mean()

    @property
    def accuracy(self):
        correct = (self.y.argmax(axis=1) == self.t.argmax(axis=1))
        return correct.sum() / correct.size

    @property
    def sample_size(self):
        return len(self.t)

    def initialize(self):
        for layer in self.layers[1:]:
            layer.w = np.random.randn(*layer.w.shape)

    def feed_data(self, x, t):
        self.x = x
        self.t = t
        return self

    def update(self, eta, mu=0):
        self.fprop()
        self.bprop()
        for layer in reversed(self.dense_layers):
            layer.w += - eta * layer.grad

    def fprop(self):
        self.input_layer.y = self.x
        for layer in self.dense_layers:
            layer.a = layer.prev.y @ layer.w
            layer.y = layer.f()
        self.y = self.output_layer.y

    def bprop(self):
        layer = self.output_layer
        delta = self.output_layer.output_delta(self.t)
        layer.grad = layer.prev.y.T @ delta / self.sample_size
        for layer in reversed(self.hidden_layers):
            delta = (delta @ layer.next.w.T) * layer.f_prime()
            layer.grad = layer.prev.y.T @ delta / self.sample_size


class Network(NaiveNetwork):
    _hidden_layer_class = SigmoidLayer

    def initialize(self):
        for layer in self.layers[1:]:
            layer.w = np.random.randn(*layer.w.shape)
            layer.v = np.zeros(layer.w.shape)

    def update(self, eta, mu=0):
        self.fprop()
        self.bprop()
        for layer in reversed(self.hidden_layers):
            layer.v = mu * layer.v - eta * layer.grad
            layer.w += layer.v
