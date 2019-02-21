import gc
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from argparse import ArgumentParser

from tsfresh.feature_extraction import extract_features
from tqdm import tqdm


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
        feats = extract_features(
            parq,
            default_fc_parameters=fc_parameters,
            column_id="dummy",
            column_kind="group",
            column_value=f"phase{i}",
            n_jobs=n_jobs)
        for j in range(ncols):
            feats_per_cols = feats.filter(
                regex=f"^{float(j)}__").values.reshape((-1, n_dims, n_feats))
            feat_mat[j, :, i * n_feats:(i + 1) * n_feats] = feats_per_cols
    return feat_mat


class Transformer:
    def __init__(self,
                 logger,
                 sample_size=800000,
                 n_dim=160,
                 min_max=(-1, 1),
                 min_num=-128,
                 max_num=127):
        self.n_dim = n_dim
        self.min_max = min_max
        self.min_num = min_num
        self.max_num = max_num
        self.sample_size = sample_size
        self.logger = logger

        self.bucket_size = int(sample_size / n_dim)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metadata")
    parser.add_argument("--path", default="../input/train.parquet")
    parser.add_argument("--name", default="../input/train_feats.pkl")
    parser.add_argument("--nchunk", default=24, type=int)
    args = parser.parse_args()
    feats_list = []

    meta = pd.read_csv(args.metadata)
    n_line = int(meta.shape[0] // 3)
    if n_line % args.nchunk == 0:
        nchunk = args.nchunk
    else:
        nchunk = args.nchunk + 1
    step = n_line // args.nchunk
    current_head = meta.signal_id[0] 
    for i in range(nchunk):
        if i == nchunk - 1:
            step = n_line % args.nchunk
        feats = fresh_features(
            path=args.path,
            ncols=step,
            offset=current_head,
            n_jobs=8,
            fc_parameters={
                "fft_coefficient": [
                    {"coeff": 0, "attr": "abs"},
                    {"coeff": 1, "attr": "abs"},
                    {"coeff": 2, "attr": "abs"}
                ],
                'longest_strike_above_mean': None,
                'longest_strike_below_mean': None,
                'mean_change': None,
                'mean_abs_change': None,
                'mean': None,
                'maximum': None,
                'minimum': None,
                'absolute_sum_of_changes': None,
                'autocorrelation': [{'lag': 3}],
                'binned_entropy': [{'max_bins': 10}],
                'cid_ce': [{'normalize': True}],
                'count_above_mean': None,
                'first_location_of_maximum': None,
                'first_location_of_minimum': None,
                'last_location_of_maximum': None,
                'last_location_of_minimum': None,
                'mean_second_derivative_central': None,
                'median': None,
                'ratio_beyond_r_sigma': [{'r': 2}],
                'time_reversal_asymmetry_statistic': [{'lag': 4}],
                "abs_energy": None,
                "kurtosis": None,
                "skewness": None,
                "standard_deviation": None,
                "sum_values": None
            })
        feats_list.append(feats)
        current_head += i * step
    with open(args.name, "wb") as f:
        pickle.dump(feats_list, f)
