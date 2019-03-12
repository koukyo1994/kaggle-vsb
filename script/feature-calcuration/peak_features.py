import numpy as np


def detrend(signal):
    x = np.diff(signal, n=1)
    return x
