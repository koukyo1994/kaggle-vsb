import gc
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features
from tqdm import tqdm

from denoising import highpass_wavelet, flatiron


def fresh_features(path="../input/train.parquet",
                   offset=0,
                   ncols=1452,
                   n_dims=160,
                   n_jobs=2,
                   fc_parameters={}):
    nrows = 800000 * ncols
    pq_mat = np.zeros((nrows, 3))
    bucket_size = int(800000 / n_dims)

    idx = np.repeat(range(n_dims), bucket_size)
    dummy = np.tile(idx, ncols).reshape(-1, 1)
    group = np.repeat(range(ncols), 800000).reshape(-1, 1)
    pq_mat = np.concatenate([pq_mat, dummy, group], axis=1)
    del dummy, group
    gc.collect()

    temp = pq.read_pandas(
        path, columns=[str(offset),
                       str(offset + 1),
                       str(offset + 2)]).to_pandas().values
    pq_mat[0:800000, 0:3] = temp

    for i in tqdm(range(1, ncols)):
        temp = pq.read_pandas(
            path,
            columns=[
                str(offset + i),
                str(offset + i + 1),
                str(offset + i + 2)
            ]).to_pandas().values
        pq_mat[i * 800000:(i + 1) * 800000, 0:3] = temp
    del temp
    gc.collect()
    parq = pd.DataFrame(
        pq_mat, columns=["phase0", "phase1", "phase2", "dummy", "group"])
    n_feats = 0
    for v in fc_parameters.values():
        if isinstance(v, list):
            n_feats += len(v)
        else:
            n_feats += 1
    feat_mat = np.zeros((ncols, n_dims, 3 * n_feats))
    for i in range(3):
        ss = StandardScaler()
        feats = extract_features(
            parq,
            default_fc_parameters=fc_parameters,
            column_id="dummy",
            column_kind="group",
            column_value=f"phase{i}",
            n_jobs=n_jobs)
        columns = feats.columns
        feats = ss.fit_transform(feats)
        feats = pd.DataFrame(feats, columns=columns)

        for j in range(ncols):
            feats_per_cols = feats.filter(
                regex=f"^{float(j)}__").values.reshape((-1, n_dims, n_feats))
            feat_mat[j, :, i * n_feats:(i + 1) * n_feats] = feats_per_cols
    return feat_mat


def _min_max_transf(ts, min_data, max_data, range_needed=(-1, 1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data + abs(min_data))
    if range_needed[0] < 0:
        return ts_std * (
            range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def _transform_ts(ts, n_dim=160, min_max=(-1, 1)):
    ts_std = _min_max_transf(ts, min_data=-128, max_data=127)
    bucket_size = int(800000 / n_dim)
    new_ts = []
    for i in range(0, 800000, bucket_size):
        ts_range = ts_std[i:i + bucket_size]
        mean = ts_range.mean()
        std = ts_range.std()
        std_top = mean + std
        std_bot = mean - std
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]
        relative_percentile = percentil_calc - mean
        new_ts.append(
            np.concatenate([
                np.asarray([mean, std, std_top, std_bot, max_range]),
                percentil_calc, relative_percentile
            ]))
    return np.asarray(new_ts)


def _calculator(ts, i, bucket_size):
    ts_range: np.ndarray = ts[i:i + bucket_size]
    mean = ts_range.mean()
    std = ts_range.std()
    std_top = mean + std
    std_bot = mean - std
    percentile_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
    max_range = percentile_calc[-1] - percentile_calc[0]
    relative_percentile = percentile_calc - mean
    return np.concatenate([
        np.asarray([mean, std, std_top, std_bot, max_range]), percentile_calc,
        relative_percentile
    ])


def _trns_after_denoising(ts, hp=True, dn=True, n_dim=160, min_max=(-1, 1)):
    ts_hp, ts_dn = highpass_wavelet(ts)
    assert hp or dn
    if hp:
        ts_hp_std = _min_max_transf(ts_hp, min_data=-128, max_data=127)
    if dn:
        ts_dn_std = _min_max_transf(ts_dn, min_data=-128, max_data=127)

    bucket_size = int(800000 / n_dim)
    new_ts = []
    for i in range(0, 800000, bucket_size):
        if hp:
            hp_feats = _calculator(ts_hp_std, i, bucket_size)
        if dn:
            dn_feats = _calculator(ts_dn_std, i, bucket_size)
        if hp and dn:
            new_ts.append(np.concatenate([hp_feats, dn_feats]))
        elif hp:
            new_ts.append(hp_feats)
        elif dn:
            new_ts.append(dn_feats)
    return np.asarray(new_ts)


def _median_transform(ts, n_dim=160, min_max=(-1, 1)):
    ts_std = _min_max_transf(ts, min_data=-128, max_data=127)
    bucket_size = int(800000 / n_dim)
    new_ts = []
    for i in range(0, 800000, bucket_size):
        ts_range = ts_std[i:i + bucket_size]
        median = ts_range.median()
        std = ts_range.std()
        std_top = median + std
        std_bot = median + std
        percentile_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
        max_range = percentile_calc[-1] - percentile_calc[0]
        relative_percentile = percentile_calc - median
        new_ts.append(
            np.concatenate([
                np.asarray([median, std, std_top, std_bot, max_range]),
                percentile_calc, relative_percentile
            ]))
    return np.asarray(new_ts)


def prep_data(path="../input/train.parquet",
              func=_transform_ts,
              offset=0,
              ncols=1452,
              n_dim=160):
    praq_train = pq.read_pandas(
        path,
        columns=[str(i) for i in range(offset, offset + ncols)]).to_pandas()

    X = []
    for i in tqdm(range(offset, offset + ncols, 3)):
        X_signal = []
        for phase in [0, 1, 2]:
            trns = func(praq_train[str(i + phase)], n_dim=n_dim)
            X_signal.append(trns)
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)
    return X


def lgbm_prep_data(path="../input/train.parquet",
                   func=_transform_ts,
                   offset=0,
                   ncols=1452,
                   n_dim=160):
    praq_train = pq.read_pandas(
        path,
        columns=[str(i) for i in range(offset, offset + ncols)]).to_pandas()

    X = []
    for i in tqdm(range(offset, offset + ncols, 3)):
        X_signal = []
        for phase in [0, 1, 2]:
            trns = func(praq_train[str(i + phase)], n_dim=n_dim)

            X_signal.append(trns)
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)
    return X


def square_data(path="../features/basic-features/160d/train_basic.pkl",
                scalers=None):
    with open(path, "rb") as f:
        array = pickle.load(f)
    for col in range(array.shape[2]):
        array[:, :, col] = array[:, :, col]**2
    if scalers:
        for row in range(array.shape[1]):
            array[:, row, :] = scalers[row].transform(array[:, row, :])
    else:
        scalers = {}
        for row in range(array.shape[1]):
            scalers[row] = StandardScaler()
            array[:, row, :] = scalers[row].fit_transform(array[:, row, :])
    return scalers, array


def feature_calculator(ts, func=np.mean, n_dim=160):
    bucket_size = int(800000 / n_dim)
    new_ts = []
    for i in range(0, 800000, bucket_size):
        ts_range = ts[i:i + bucket_size]
        if isinstance(func, list):
            for f in func:
                new_ts.append(f(ts_range))
        else:
            new_ts.append(func(ts_range))
    return np.asarray(new_ts).reshape((n_dim, -1))


def prep_data_feature_wise(path="../input/train.parquet",
                           func=[],
                           offset=0,
                           ncols=1452,
                           n_dim=160):
    praq_train = pq.read_pandas(
        path,
        columns=[str(i) for i in range(offset, offset + ncols)]).to_pandas()
    X = []
    for i in tqdm(range(offset, offset + ncols, 3)):
        X_signal = []
        for phase in [0, 1, 2]:
            trns = feature_calculator(
                praq_train[str(i + phase)], func, n_dim=n_dim)
            X_signal.append(trns)
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)
    return X


def prep_data_denoising(path="../input/train.parquet",
                        offset=0,
                        ncols=1452,
                        n_dim=160,
                        hp=True,
                        dn=True):
    praq_train = pq.read_pandas(
        path,
        columns=[str(i) for i in range(offset, offset + ncols)]).to_pandas()
    praq_columns = praq_train.columns
    data = flatiron(praq_train.values)
    praq_train = pd.DataFrame(data=data, columns=praq_columns)
    X = []
    for i in tqdm(range(offset, offset + ncols, 3)):
        X_signal = []
        for phase in [0, 1, 2]:
            trns = _trns_after_denoising(
                praq_train[str(i + phase)], n_dim=n_dim, hp=hp, dn=dn)
            X_signal.append(trns)
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)
    return X
