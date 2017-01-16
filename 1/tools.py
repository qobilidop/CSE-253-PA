import matplotlib.pyplot as plt
import pandas as pd


class TrainingResults(object):
    def __init__(self, records, epoch_final):
        self.records = records
        self.epoch_final = epoch_final

    def plot_loss_function(self):
        plt.figure()
        plt.plot(self.records['E_train'], label='train')
        plt.plot(self.records['E_hold'], label='hold-out')
        plt.plot(self.records['E_test'], label='test')
        plt.axvline(self.epoch_final, color='gray',
                    label='final epoch {}'.format(self.epoch_final))
        plt.legend(loc=0)
        plt.xlabel('epoch')
        plt.ylabel('loss function')
        plt.axvline()

    def plot_percent_correct(self):
        plt.figure()
        plt.plot(self.records['c_train'], label='train')
        plt.plot(self.records['c_hold'], label='hold-out')
        plt.plot(self.records['c_test'], label='test')
        plt.axvline(self.epoch_final, color='gray',
                    label='final epoch {}'.format(self.epoch_final))
        plt.legend(loc=0)
        plt.xlabel('epoch')
        plt.ylabel('percent correct')

    def plot_final_weights(self):
        weights = self.records['weights'][epoch_final]
        weights =


def train(nn_class, dss, rate, lam=0, epoch_min=None, epoch_max=None,
          early_stopping=3):
    nn = nn_class(dss.train.dim)
    records = pd.DataFrame()
    epoch = 0

    while True:
        record = pd.Series()

        nn.use_labeled_images(dss.train)
        nn.update(rate(epoch), lam)
        record['weights'] = nn.weights
        record['E_train'] = nn.loss_function()
        record['c_train'] = nn.percent_correct()

        nn.use_labeled_images(dss.validation)
        record['E_hold'] = nn.loss_function()
        record['c_hold'] = nn.percent_correct()

        nn.use_labeled_images(dss.test)
        record['E_test'] = nn.loss_function()
        record['c_test'] = nn.percent_correct()

        records = records.append(record, ignore_index=True)
        epoch += 1

        if (epoch_max is not None) and (epoch > epoch_max):
            break

        if (epoch_min is None) or (epoch > epoch_min):
            if early_stopping is not None:
                s = early_stopping
                consistent_increase = (records['E_hold'][-(s + 1):]
                                       .diff()[-s:] > 0).all()
                if consistent_increase:
                    break

    epoch_final = records['E_hold'].argmin()
    return TrainingResults(records, epoch_final)
