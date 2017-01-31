import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_set_labels = ['training', 'validation', 'test']


class TrainingResult(object):
    def __init__(self, best_model, history, final_epoch):
        self.best_model = best_model
        self.history = history
        self.final_epoch = final_epoch

    def plot_history(self, quantity):
        plt.figure()
        for label in data_set_labels:
            plt.plot(self.history['epoch'],
                     self.history['_'.join([quantity, label])],
                     label=label)
        plt.legend(loc=0)
        plt.xlabel('Epoch')
        plt.ylabel(quantity.capitalize())


def train(network, dss, update_params,
          epoch_min=None, epoch_max=None, early_stopping=3):
    history = pd.DataFrame()
    loss_min = np.inf
    best_model = None
    final_epoch = None
    epoch = 0
    network.initialize()
    while True:
        record = pd.Series()
        record['epoch'] = epoch
        network.feed_data(dss.training.images, dss.training.labels)
        network.update(**update_params)

        for label in data_set_labels:
            ds = getattr(dss, label)
            network.feed_data(ds.images, ds.labels)
            if label != 'training':
                network.fprop()
            record['_'.join(['loss', label])] = network.loss
            record['_'.join(['accuracy', label])] = network.accuracy

        if record['loss_validation'] < loss_min:
            loss_min = record['loss_validation']
            best_model = copy.deepcopy(network)
            final_epoch = epoch

        history = history.append(record, ignore_index=True)
        epoch += 1

        if (epoch_max is not None) and (epoch > epoch_max):
            break

        if (epoch_min is None) or (epoch > epoch_min):
            if early_stopping:
                s = early_stopping
                consistent_increase = (history['loss_validation'][-(s + 1):]
                                       .diff()[-s:] > 0).all()
                if consistent_increase:
                    break

    return TrainingResult(best_model, history, final_epoch)
