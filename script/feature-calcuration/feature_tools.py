import numpy as np
import pyarrow.parquet as pq

from tqdm import tqdm

from denoising import highpass_wavelet


def _outlier_robust_calculator(ts, i, bucket_size):
    ts_range: np.ndarray = ts[i:i + bucket_size]
    median = np.median(ts_range)
    std = ts_range.std()
    std_top = median + std
    std_bot = median - std
    percentile_calc = np.percentile(ts_range, [10, 25, 50, 75, 90])
    max_range = percentile_calc[-1] - percentile_calc[0]
    relative_percentile = percentile_calc - median
    return np.concatenate([
        np.asarray([median, std, std_top, std_bot, max_range]),
        percentile_calc, relative_percentile
    ])


def _trns_after_denoising(ts, n_dim=160):
    _, ts_dn = highpass_wavelet(ts)
    bucket_size = int(800000 / n_dim)
    new_ts = []
    for i in range(0, 800000, bucket_size):
        dn_feats = _outlier_robust_calculator(ts_dn, i, bucket_size)
        new_ts.append(dn_feats)
    return np.asarray(new_ts)


def robust_denoised_data(path="../input/train.parquet",
                         offset=0,
                         ncols=1452,
                         n_dim=160):
    parq_train = pq.read_pandas(
        path,
        columns=[str(i) for i in range(offset, offset + ncols)]).to_pandas()
    X = []
    for i in tqdm(range(offset, offset + ncols, 3)):
        X_signal = []
        for phase in [0, 1, 2]:
            trns = _trns_after_denoising(
                parq_train[str(i + phase)], n_dim=n_dim)
            X_signal.append(trns)
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)
    return X
