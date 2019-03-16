import sys
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")

    from lgbm_features_extractor import signal_origins, prep_features
    from feature_extraction import flatiron
    from script.common.utils import timer, get_logger

    parser = ArgumentParser()
    parser.add_argument("--metadata", default="../input/metadata_train.csv")
    parser.add_argument("--parquet", default="../input/train.parquet")
    parser.add_argument("--name", default="train.pkl")
    parser.add_argument("--nchunk", default=2, type=int)

    args = parser.parse_args()
    outdir = Path("../features/lgbm-features")
    outdir.mkdir(exist_ok=True, parents=True)

    logger = get_logger("lgbm", tag="lgbm")
    logger.info(f"parquet: {args.parquet}")

    metadata = pd.read_csv(args.metadata)
    n_line = int(metadata.shape[0] // 3)
    if n_line % args.nchunk == 0:
        nchunk = args.nchunk
    else:
        nchunk = args.nchunk + 1
    step = (n_line // args.nchunk) * 3
    current_head = metadata.signal_id[0]

    logger.info(f"n_line: {n_line}")
    logger.info(f"nchunk: {nchunk}")
    logger.info(f"step: {step}")
    logger.info(f"current_head: {current_head}")
    X = []

    for i in range(nchunk):
        if (i == nchunk - 1) and (n_line % args.nchunk != 0):
            step = metadata.signal_id.max() - current_head + 1
        with timer(f"chunk{i+1}", logger):
            parq = pq.read_pandas(
                args.parquet,
                columns=[
                    str(i) for i in range(current_head, current_head + step)
                ],
                nthreads=-1).to_pandas().values
            origins = signal_origins(parq)

            data = flatiron(parq)
            X_temp = prep_features(data, origins)
            X.append(X_temp)
            current_head += step
            logger.info(f"current_head: {current_head}")
    X = np.concatenate(X)
    logger.info(f"X_shape: {X.shape}")
    with open(outdir / args.name, "wb") as f:
        pickle.dump(X, f)
