import numpy as np
import pywt

from scipy import cluster
from scipy import stats
from scipy.signal import (find_peaks, peak_widths, peak_prominences, sosfilt,
                          butter)
from tqdm import tqdm

from joblib import Parallel, delayed


def find_zero_crossing(signal: np.ndarray):
    return np.where(np.diff(np.sign(signal)))[0]


def is_neg_to_pos_crossing(signal: np.ndarray, candidate: int):
    i = candidate
    before: np.ndarray = signal.take(range(i - 10000, i), mode="wrap")
    after: np.ndarray = signal.take(range(i, i + 10000), mode="wrap")
    return before.mean() < 0 and after.mean() > 0


def find_origin(signal: np.ndarray):
    crossings = find_zero_crossing(signal)
    a, b = cluster.vq.kmeans(
        crossings.astype(float), 2, iter=2, thresh=1e-5)[0]
    a, b = int(a), int(b)
    if is_neg_to_pos_crossing(signal, a):
        return a
    else:
        return b


def mad(x, axis=None):
    return np.mean(np.abs(x - np.mean(x, axis)), axis)


def wavelet_denoise(x, wavelet="db1", mode="hard"):
    c_a, c_d = pywt.dwt(x, wavelet)
    sigma = 1 / 0.6745 * mad(np.abs(c_d))
    threshold = sigma * np.sqrt(2 * np.log(len(x)))
    c_d_t = pywt.threshold(c_d, threshold, mode=mode)
    y = pywt.idwt(np.zeros_like(c_a), c_d_t, wavelet)
    return y


def highpass_filter(x, low_cutoff=1000):
    n_samples = 800000
    sample_duration = 0.02
    sample_rate = n_samples * 1 / sample_duration

    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist

    sos = butter(10, Wn=[norm_low_cutoff], btype="highpass", output="sos")
    filtered_signal = sosfilt(sos, x)
    return filtered_signal


def signal_origins(signals: np.ndarray):
    origins = []
    for i in tqdm(range(signals.shape[1])):
        signal: np.ndarray = signals[:, i]
        origins.append(find_origin(signal))
    return np.array(origins)


def compute_features(signal: np.ndarray, origin: int, funcs: list):
    signal = highpass_filter(signal, 1e4)
    signal = wavelet_denoise(signal)
    signal = np.roll(signal, 800000 - origin)
    features = []
    for func in funcs:
        features += func(signal)
    return features


def peaks(signal: np.ndarray):
    peaks, properties = find_peaks(signal)
    width = peak_widths(signal, peaks)[0]
    prominence = peak_prominences(signal, peaks)[0]
    return [
        peaks.size,
        width.mean() if width.size else -1.,
        width.max() if width.size else -1.,
        width.min() if width.size else -1.,
        prominence.mean() if prominence.size else -1.,
        prominence.max() if prominence.size else -1.,
        prominence.min() if prominence.size else -1.
    ]


def signal_entropy(signal: np.ndarray):
    sig = signal.copy()

    for i in range(3):
        max_pos = sig.argmax()
        sig[max_pos - 1000:max_pos + 1000] = 0.
    return [stats.entropy(np.histogram(sig, 15)[0])]


def bucketed_entropy(signal: np.ndarray):
    sig = signal.copy()
    return [stats.entropy(np.histogram(y, 10)[0]) for y in np.split(sig, 10)]


def prep_features(signals: np.ndarray, origins: np.ndarray):
    funcs = [peaks, signal_entropy, bucketed_entropy]
    sig_feat = Parallel(
        n_jobs=-1, verbose=1)([
            delayed(compute_features)(signals[:, i], origin, funcs)
            for i, origin in zip(range(signals.shape[1]), origins)
        ])
    return np.array(sig_feat)
