import sys
import pickle
import numpy as np
import pandas as pd

from pathlib import Path

from argparse import ArgumentParser

if __name__ == '__main__':
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")

    from script.common.trainer import LGBMTrainer
    from script.common.utils import get_logger

    parser = ArgumentParser()
    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--enable_local_test", action="store_true")
    parser.add_argument("--test_size", default=0.3, type=float)
    parser.add_argument("--n_epochs", default=5000, type=int)

    parser.add_argument("--num_leaves", default=31, type=int)
    parser.add_argument("--learning_rate", default=0.15, type=float)
    parser.add_argument("--min_child_weight", default=1e-3, type=float)
    parser.add_argument("--subsample", default=0.8, type=float)
    parser.add_argument("--subsample_freq", default=5, type=int)
    parser.add_argument("--colsample_bytree", default=0.8, type=float)
    parser.add_argument("--reg_alpha", default=0.01, type=float)
    parser.add_argument("--reg_lambda", default=0.01, type=float)
    parser.add_argument("--n_jobs", default=2, type=int)

    parser.add_argument("--metadata")
    parser.add_argument("--features")

    args = parser.parse_args()

    logger = get_logger("lightgbm", "lightgbm")

    logger.info(
        f"num_leaves: {args.num_leaves}, learning_rate: {args.learning_rate}")
    logger.info(f"min_child_weight: {args.min_child_weight}")
    logger.info(
        f"subsample: {args.subsample}, subsample_freq: {args.subsample_freq}")
    logger.info(f"colsample_bytree: {args.colsample_bytree}")
    logger.info(f"reg_alpha: {args.reg_alpha}, reg_lambda: {args.reg_lambda}")
    logger.info(f"n_splits: {args.n_splits}, seed: {args.seed}")
    logger.info(f"n_epochs: {args.n_epochs}")

    with open(args.features, "rb") as f:
        train = pickle.load(f)

    answer = pd.read_csv(args.metadata).target.values
    trainer = LGBMTrainer(
        logger,
        n_splits=args.n_splits,
        seed=args.seed,
        enable_local_test=args.enable_local_test,
        test_size=args.test_size,
        kwargs={
            "num_leaves": args.num_leaves,
            "learning_rate": args.learning_rate,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "subsample_freq": args.subsample_freq,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "n_jobs": args.n_jobs
        })
    trainer.fit(train, answer, args.n_epochs)
    trainer_path = Path(f"trainer/{trainer.tag}")
    trainer_path.mkdir(parents=True, exist_ok=True)
