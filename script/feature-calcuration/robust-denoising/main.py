import sys
import pickle
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")

    from script.common.utils import timer, get_logger
    from feature_tools import robust_denoised_data

    parser = ArgumentParser()
    parser.add_argument("--n_dims", default=160, type=int)

    args = parser.parse_args()
    outdir = Path(f"../features/robust-denoising/{args.n_dims}")
    outdir.mkdir(exist_ok=True, parents=True)

    logger = get_logger(
        name="robust-denoising", tag=f"robust-denoising/{args.n_dims}")
    meta_train = pd.read_csv("../input/metadata_train.csv")
    meta_test = pd.read_csv("../input/metadata_test.csv")

    train_path = Path("../input/train.parquet")
    test_path = Path("../input/test.parquet")

    n_line = int(meta_train.shape[0] // 3)
    nchunk_train = 2
    step = (n_line // nchunk_train) * 3
    current_head = meta_train.signal_id[0]
    logger.info(f"step: {step}")
    logger.info(f"initial head: {current_head}")
    X = []
    for i in range(nchunk_train):
        with timer(f"chunk{i+1}", logger):
            X_temp = robust_denoised_data(train_path, current_head, step,
                                          args.n_dims)
            X.append(X_temp)
            current_head += step
            logger.info(f"current head: {current_head}")
    X = np.concatenate(X)
    logger.info(f"X_shape: {X.shape}")
    with open(outdir / "train.pkl", "wb") as f:
        pickle.dump(X, f)

    n_line = int(meta_test.shape[0] // 3)
    nchunk_test = 7
    step = (n_line // 6) * 3
    current_head = meta_test.signal_id[0]
    logger.info(f"step: {step}")
    logger.info(f"initial head: {current_head}")
    X = []
    for i in range(nchunk_test):
        if (i == nchunk_test - 1):
            step = meta_test.signal_id.max() - current_head + 1
        with timer(f"chunk{i+1}", logger):
            X_temp = robust_denoised_data(test_path, current_head, step,
                                          args.n_dims)
            X.append(X_temp)
            current_head += step
            logger.info(f"current head: {current_head}")
    X = np.concatenate(X)

    logger.info(f"X_shape: {X.shape}")
    with open(outdir / "test.pkl", "wb") as f:
        pickle.dump(X, f)
