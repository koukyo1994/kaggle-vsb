import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path


class CycleAugmentor:
    def __init__(self, path, meta_path):
        self.path = Path(path)
        self.meta_path = Path(meta_path)

        self.meta = pd.read_csv(self.meta_path)

        self.n_line = self.meta.shape[0]
        self.n_pair = int(self.n_line // 3)
        self.augmented_data = None

    def __dump__(self, path):
        assert self.augmented_data
        self.augmented_data.to_parquet(path)

    def augment(self, save_path, n_original=100, n_new_per_col=2, seed=1213):
        np.random.seed = seed
        idx_list = np.arange(self.n_pair)
        selected_idx = np.sort(
            np.random.choice(idx_list, n_original, replace=False) * 3)
        selected_cols = [str(i + j) for i in selected_idx for j in range(3)]
        parq = pq.read_pandas(self.path, columns=selected_cols, nthreads=-1)

        ts_list = []
        for i in range(len(selected_cols), step=3):
            new_ts = np.zeros((800000, 3))
