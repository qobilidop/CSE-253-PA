import numpy as np

class Network(object):
    def __init__(self, size):
        self.w = [np.zeros((i, j)) for i, j in zip(size[:-1], size[1:])]
        self.size = size

    @property
    def L(self):
        return len(self.size)

    def initialize(self):
        self.w = [np.random.normal(scale=1 / np.sqrt(i), size=(i, j))
                  for i, j in zip(self.size[:-1], self.size[1:])]

    def loss(self, x, t):
        a, y = self.feedforward(x)
        return -(t * np.log(y[self.L - 1])).sum(axis=1).mean()

    def update(self, x, t, rate):
        a, y = self.feedforward(x)
        w_grad = self.backprop(a, y, t)
        for l in reversed(range(self.L - 1)):
            self.w[l] -= rate * w_grad[l]

    def sigmoid(self, z):
        large = 100
        return 1.0 / (1.0 + np.exp(np.minimum(-z, large)))

    def prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, x):
        a = [None] * self.L
        y = [None] * self.L
        y[0] = x  # N * Di
        for l in range(self.L - 2):
            a[l + 1] = y[l] @ self.w[l]  # N * Dj
            y[l + 1] = self.sigmoid(a[l + 1])  # N * Dj
        a[self.L - 1] = y[self.L - 2] @ self.w[self.L - 2]
        large = 100
        exp = np.exp(np.minimum(a[self.L - 1], large))
        #print(exp.sum(axis=1))
        y[self.L - 1] = exp / np.expand_dims(exp.sum(axis=1), axis=1)
        #print(y[self.L - 1])
        return a, y

    def backprop(self, a, y, t):
        delta =  [None] * (self.L)
        w_grad = [None] * (self.L - 1)
        delta[self.L - 1] = - t + y[-1] # N * DL
        w_grad[self.L - 2] = y[self.L - 2].T @ delta[self.L - 1]
        #print(w_grad[self.L-2].shape)
        for l in reversed(range(1, self.L - 1)):
            delta[l] = (delta[l + 1] @ self.w[l].T) * self.prime(a[l])
            w_grad[l - 1] = y[l - 1].T @ delta[l]  # Fix matmul
        return w_grad

    def percent_correct(self, x, t):
        a, y = self.feedforward(x)
        correct = (t.argmax(axis=1) == y[self.L - 1].argmax(axis=1))
        return correct.sum() / correct.size