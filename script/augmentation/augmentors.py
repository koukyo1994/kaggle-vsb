import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


class ShakingAugmentor:
    def __init__(self, path, meta_path):
        self.path = Path(path)
        self.meta_path = Path(meta_path)

        self.meta: pd.DataFrame = pd.read_csv(self.meta_path)

        self.n_line: int = self.meta.shape[0]
        self.n_pair = int(self.n_line // 3)
        self.augmented_data = None

        self.pos_idx: list = self.meta.query(
            "phase == 0 & target == 1").target.index.tolist()
        self.neg_idx: list = self.meta.query(
            "phase == 0 & target == 0").target.index.tolist()

    def __create_new_ts(self, ts):
        new_ts = np.zeros((800000, 3))
        cut_off_idx = int(np.random.rand() * 800000)
        ts_head = ts[cut_off_idx:, :]
        ts_tail = ts[:cut_off_idx, :]
        new_ts[:(800000 - cut_off_idx), :] = ts_head
        new_ts[(800000 - cut_off_idx):, :] = ts_tail
        return new_ts

    def augment(self, n_pos=100, n_neg=100, n_new_per_col=2, seed=1213):
        np.random.seed = seed
        if n_pos > len(self.pos_idx):
            n_pos = len(self.pos_idx)

        if n_neg > len(self.neg_idx):
            n_neg = len(self.neg_idx)

        selected_pos_idx = np.sort(
            np.random.choice(self.pos_idx, n_pos, replace=False))
        selected_neg_idx = np.sort(
            np.random.choice(self.neg_idx, n_neg, replace=False))
        selected_pos_cols = [
            str(i + j) for i in selected_pos_idx for j in range(3)
        ]
        selected_neg_cols = [
            str(i + j) for i in selected_neg_idx for j in range(3)
        ]
        parq_pos: pd.DataFrame = pq.read_pandas(
            self.path, columns=selected_pos_cols, nthreads=-1).to_pandas()
        parq_neg: pd.DataFrame = pq.read_pandas(
            self.path, columns=selected_neg_cols, nthreads=-1).to_pandas()

        ts_augmented = np.zeros([800000, (n_neg + n_pos) * n_new_per_col * 3])
        current_head = 0
        label_list = []
        for i in tqdm(selected_pos_cols[::3]):
            ts: np.ndarray = parq_pos.loc[:, [
                i, str(int(i) + 1), str(int(i) + 2)
            ]].values
            for _ in range(n_new_per_col):
                new_ts = self.__create_new_ts(ts)
                ts_augmented[:, current_head:current_head + 3] = new_ts
                label_list += [1, 1, 1]
                current_head += 3

        for i in tqdm(selected_neg_cols[::3]):
            ts: np.ndarray = parq_neg.loc[:, [
                i, str(int(i) + 1), str(int(i) + 2)
            ]].values
            for _ in range(n_new_per_col):
                new_ts = self.__create_new_ts(ts)
                ts_augmented[:, current_head:current_head + 3] = new_ts
                label_list += [0, 0, 0]
                current_head += 3

        columns = np.arange(
            start=self.n_line, stop=self.n_line + ts_augmented.shape[1])
        str_columns = [str(c) for c in columns]
        ts_augmented = pd.DataFrame(data=ts_augmented, columns=str_columns)
        last_id = self.meta.id_measurement.max()
        n_new_id_measurement = ts_augmented.shape[1] // 3
        new_id_measurement = [
            last_id + 1 + i for i in range(n_new_id_measurement)
            for _ in range(3)
        ]
        new_meta = pd.DataFrame({
            "signal_id":
            columns,
            "id_measurement":
            new_id_measurement,
            "phase":
            [i for _ in range(n_new_id_measurement) for i in range(3)],
            "target":
            label_list
        })
        new_meta.index = pd.Index(columns)
        return ts_augmented, new_meta


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--save_dir", default="../augmented-inputs")
    parser.add_argument("--n_pos", default=10, type=int)
    parser.add_argument("--n_neg", default=10, type=int)
    parser.add_argument("--n_new_per_col", default=2, type=int)
    parser.add_argument("--seed", default=1213, type=int)

    args = parser.parse_args()
    aug = ShakingAugmentor("../input/train.parquet",
                           "../input/metadata_train.csv")
    new_ts, new_meta = aug.augment(
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        n_new_per_col=args.n_new_per_col,
        seed=args.seed)

    path = Path(args.save_dir)
    path.mkdir(exist_ok=True)

    new_ts.to_parquet(
        path /
        f"augmented_{args.n_pos}_{args.n_neg}_{args.n_new_per_col}.parquet")
    new_meta.to_csv(
        path / f"augmented_{args.n_pos}_{args.n_neg}_{args.n_new_per_col}.csv",
        index=False)
