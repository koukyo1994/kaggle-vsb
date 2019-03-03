import sys
import pickle
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("..")
    sys.path.append("./")

    from script.common.utils import timer, get_logger
    from feature_extraction import prep_data, _transform_ts

    parser = ArgumentParser()
    parser.add_argument("--metadata")
    parser.add_argument("--path", default="../input/train.parquet")
    parser.add_argument("--name", default="train_basic.pkl")
    parser.add_argument("--n_dims", default=160, type=int)
    parser.add_argument("--nchunk", default=2, type=int)

    args = parser.parse_args()
    outdir = Path(f"../features/basic-features/{args.n_dims}d/")
    outdir.mkdir(exist_ok=True, parents=True)

    logger = get_logger(name="basic", tag=f"basic_features/{args.n_dims}d")
    logger.info(f"path: {args.path}, name: {args.name}")

    meta = pd.read_csv(args.metadata)
    n_line = int(meta.shape[0] // 3)
    logger.info(f"n_line: {n_line}")
    if n_line % args.nchunk == 0:
        nchunk = args.nchunk
    else:
        nchunk = args.nchunk + 1
    step = (n_line // args.nchunk) * 3
    current_head = meta.signal_id[0]
    logger.info(f"step: {step}")
    logger.info(f"initial head: {current_head}")
    X = []
    for i in range(nchunk):
        if (i == nchunk - 1) and (n_line % args.nchunk != 0):
            step = meta.signal_id.max() - current_head + 1
        with timer(f"chunk{i+1}", logger):
            X_temp = prep_data(args.path, _transform_ts, current_head, step,
                               args.n_dims)
            X.append(X_temp)
            current_head += step
            logger.info(f"current head: {current_head}")
    X = np.concatenate(X)

    logger.info(f"X_shape: {X.shape}")
    with open(outdir / args.name, "wb") as f:
        pickle.dump(X, f)
