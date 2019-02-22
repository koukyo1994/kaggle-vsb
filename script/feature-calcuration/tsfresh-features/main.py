import re
import sys
import json
import pickle
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    sys.path.append("..")
    sys.path.append("./")

    from common.utils import timer, get_logger, parse_dict
    from feature_extraction import fresh_features

    parser = ArgumentParser()
    parser.add_argument("--metadata")
    parser.add_argument("--path", default="../input/train.parquet")
    parser.add_argument("--name", default="train_feats.pkl")
    parser.add_argument("--nchunk", default=24, type=int)
    parser.add_argument("--n_jobs", default=2, type=int)
    parser.add_argument(
        "--parameters", default="tsfresh-features/init_parameters.json")
    args = parser.parse_args()
    feats_list = []

    with open(args.parameters) as f:
        parameters = json.load(f)

    parameters = parse_dict(parameters)
    filename = re.search(r"[a-zA-Z_]+.json$", args.parameters).group()

    outdir = Path(f"../features/{filename}")
    outdir.mkdir(exist_ok=True)

    logger = get_logger(name="tsfresh", tag=f"tsfresh_features/{filename}")
    logger.info(f"path: {args.path}, name: {args.name}, nchunk: {args.nchunk}")

    meta = pd.read_csv(args.metadata)
    n_line = int(meta.shape[0] // 3)
    logger.info(f"n_line: {n_line}")
    if n_line % args.nchunk == 0:
        nchunk = args.nchunk
    else:
        nchunk = args.nchunk + 1
    step = n_line // args.nchunk
    current_head = meta.signal_id[0]
    logger.info(f"step: {step}")
    logger.info(f"initial head: {current_head}")
    logger.info(f"parameters: {parameters}")

    for i in range(nchunk):
        if (i == nchunk - 1) and (n_line % args.nchunk != 0):
            step = n_line % args.nchunk
        with timer(f"chunk{i+1}", logger):
            feats = fresh_features(
                path=args.path,
                ncols=step,
                offset=current_head,
                n_jobs=args.n_jobs,
                fc_parameters=parameters)
        feats_list.append(feats)
        current_head += step
        logger.info(f"current head: {current_head}")
    with open(outdir / args.name, "wb") as f:
        pickle.dump(feats_list, f)
    logger.info(f"dumped {args.name}")
