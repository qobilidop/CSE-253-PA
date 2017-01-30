import numpy as np


def sane_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


def prevent_nan(f):
    def f_new(*args, **kwargs):
        return np.nan_to_num(f(*args, **kwargs))
    return f_new
