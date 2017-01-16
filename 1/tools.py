import matplotlib.pyplot as plt
import pandas as pd


class TrainingResults(object):
    def __init__(self, record, final_weights):
        self.record = record
        self.final_weights = final_weights

    def plot_loss_function(self):
        plt.figure()
        plt.plot(self.record['E_train'], label='train')
        plt.plot(self.record['E_hold'], label='hold-out')
        plt.plot(self.record['E_test'], label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss function')

    def plot_percent_correct(self):
        plt.figure()
        plt.plot(self.record['c_train'], label='train')
        plt.plot(self.record['c_hold'], label='hold-out')
        plt.plot(self.record['c_test'], label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('percent correct')


def train(NN, dss, rate, lam=0, epoch_min=100, epoch_max=None):
    nn = NN(dss.train.dim)
    record = pd.DataFrame()
    weights = []
    epoch = 0

    while True:
        row = pd.Series()

        nn.use_labeled_images(dss.train)
        nn.update(rate(epoch), lam)
        weights += [nn.weights]
        row['E_train'] = nn.loss_function()
        row['c_train'] = nn.percent_correct()

        nn.use_labeled_images(dss.validation)
        row['E_hold'] = nn.loss_function()
        row['c_hold'] = nn.percent_correct()

        nn.use_labeled_images(dss.test)
        row['E_test'] = nn.loss_function()
        row['c_test'] = nn.percent_correct()

        record = record.append(row, ignore_index=True)
        epoch += 1

        if (epoch_max is not None) and (epoch > epoch_max):
            break

        if epoch > epoch_min:
            if (record['E_hold'][-4:].diff()[-3:] > 0).all():
                break

    return TrainingResults(record, weights[record['E_hold'].argmin()])
