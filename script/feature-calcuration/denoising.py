import pywt
import numpy as np
import scipy.signal as signal

from scipy.signal import butter

from numba import jit, int32


@jit('float32(float32[:,:], int32, int32)')
def flatiron(x, alpha=50., beta=1):
    new_x = np.zeros_like(x)
    zero = x[0, :]
    for i in range(1, len(x)):
        x_ = x[i]
        zero = zero * 0.98 + x_ * 0.02
        new_x[i] = x_ - zero
    return new_x


def _mad_dest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def _highpass_filter(x, low_cutoff=1000):
    n_samples = 800000
    sample_duration = 0.02
    sample_rate = n_samples * 1 / sample_duration

    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist

    sos = butter(10, Wn=[norm_low_cutoff], btype="highpass", output="sos")
    filtered_signal = signal.sosfilt(sos, x)
    return filtered_signal


def _denoise(x, wavelet="db4", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * _mad_dest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard")
                 for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="per")


def highpass_wavelet(ts):
    ts_hp = _highpass_filter(ts, low_cutoff=1e4)
    ts_dn = _denoise(ts_hp, wavelet="haar")
    return ts_hp, ts_dn
