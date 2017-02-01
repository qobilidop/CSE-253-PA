import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrainingResult(object):
    data_set_labels = ['training', 'validation', 'test']

    def __init__(self):
        self.loss_min = np.inf
        self.best_model = None
        self.history = pd.DataFrame()
        self.final_epoch = None

    def capture(self, epoch, network, dss):
        record = pd.Series()
        record['epoch'] = epoch
        for label in self.data_set_labels:
            ds = getattr(dss, label)
            network.feed_data(ds.images, ds.labels).fprop()
            record['_'.join(['loss', label])] = network.loss
            record['_'.join(['accuracy', label])] = network.accuracy
        # Network with minimum loss on validation set is considered the best.
        if record['loss_validation'] < self.loss_min:
            self.loss_min = record['loss_validation']
            self.best_model = copy.deepcopy(network)
            self.final_epoch = epoch
        self.history = self.history.append(record, ignore_index=True)

    def plot_history(self, quantity):
        plt.figure()
        for label in self.data_set_labels:
            plt.plot(self.history['epoch'],
                     self.history['_'.join([quantity, label])],
                     label=label)
        plt.legend(loc=0)
        plt.xlabel('Epoch')
        plt.ylabel(quantity.capitalize())


def train(network, dss,
          eta0=None, tau=None, mu=0,
          minibatch_size=None, capture_interval=1,
          epoch_min=None, epoch_max=None, patience=10):
    if minibatch_size is None:
        minibatch_size = dss.training.size
    result = TrainingResult()
    epoch = 0
    network.initialize()
    while True:
        epoch = np.round(epoch)
        minibatches = dss.training.shuffle().minibatches(minibatch_size)
        minibatch_num = len(minibatches)
        for i, ds in enumerate(minibatches):
            eta = eta0 / (1 + epoch / tau)
            network.feed_data(ds.images, ds.labels).update(eta=eta, mu=mu)
            if i % capture_interval == 0:
                result.capture(epoch, network, dss)
            if (epoch_max is not None) and (epoch > epoch_max):
                return result
            if (epoch_min is None) or (epoch > epoch_min):
                if epoch - result.final_epoch > patience:
                    return result
            print('Epoch', epoch, end='\r')
            epoch += 1 / minibatch_num
