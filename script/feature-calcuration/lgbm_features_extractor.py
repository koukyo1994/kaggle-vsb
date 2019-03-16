import numpy as np

from scipy import cluster
from scipy import stats
from scipy.signal import find_peaks, peak_widths, peak_prominences
from tqdm import tqdm


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


def signal_origins(signals: np.ndarray):
    origins = []
    for i in tqdm(range(signals.shape[1])):
        signal: np.ndarray = signals[:, i]
        origins.append(find_origin(signal))
    return np.array(origins)


def compute_features(signal: np.ndarray, origin: int, funcs: list):
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
    sig_feat = []
    for i, origin in tqdm(zip(range(signals.shape[1]), origins)):
        features = compute_features(signals[:, i], origin, funcs)
        sig_feat.append(features)
    return np.array(sig_feat)
