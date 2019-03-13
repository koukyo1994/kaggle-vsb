import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    sys.path.append("../..")
    sys.path.append("..")
    sys.path.append("./")

    from script.common.utils import get_logger
    from vae import VAETrainer, VariationalAutoEncoder

    parser = ArgumentParser()
    parser.add_argument("--n_dim", default=5, type=int)
    parser.add_argument("--train")
    parser.add_argument("--test")

    args = parser.parse_args()
    outdir = Path("../features/vae")
    outdir.mkdir(exist_ok=True, parents=True)

    logger = get_logger(name="vae", tag="vae/basic")

    meta = pd.read_csv("../input/metadata_train.csv")
    target = meta.target.values[::3]

    submissions = os.listdir("../submission")
    sub_dfs = [pd.read_csv("../submission/" + f) for f in submissions]
    targets = [s.target.values for s in sub_dfs]
    sub_means = np.asarray(targets).mean(axis=0)
    idx = np.argwhere(sub_means == 0.0).reshape(-1)
    idx = (idx[::3] / 3).astype(int)

    with open(Path(args.train), "rb") as f:
        X = pickle.load(f)

    with open(Path(args.test), "rb") as f:
        X_test = pickle.load(f)

    X_train = X[target == 0]
    X_test_use = X_test[idx]

    fold = KFold(n_splits=5, shuffle=True, random_state=42)
    X_all = np.concatenate([X_train, X_test_use], axis=0)

    trainer = VAETrainer(
        VariationalAutoEncoder,
        logger,
        device="cpu",
        train_batch=64,
        kwargs={
            "first": 20,
            "middle": args.n_dim,
            "second": 20
        })
    for i, (trn_idx, val_idx) in enumerate(fold.split(X_all)):
        X_trn = X_all[trn_idx]
        X_val = X_all[val_idx]
        trainer.fit(X_trn, X_val, 15, i)

    X_mean = np.zeros((X.shape[0], X.shape[1], 2 * args.n_dim + 1))
    X_mean_test = np.zeros((X_test.shape[0], X_test.shape[1],
                            2 * args.n_dim + 1))
    for i in range(5):
        X_pred = trainer.predict(X, i)
        X_pred_test = trainer.predict(X_test, i)
        X_mean += X_pred / 5
        X_mean_test += X_pred_test / 5

    with open(outdir / "train.pkl", "wb") as f:
        pickle.dump(X_mean, f)

    with open(outdir / "test.pkl", "wb") as f:
        pickle.dump(X_mean_test, f)
