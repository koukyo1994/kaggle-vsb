import sys
import pickle
import numpy as np
import pandas as pd

from pathlib import Path

from argparse import ArgumentParser

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("..")
    sys.path.append("./")
    from model import LSTMGRUAttentionNet
    from script.common.trainer import NNTrainer
    from script.common.utils import get_logger

    parser = ArgumentParser()
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--linear_size", default=100, type=int)
    parser.add_argument("--n_attention", default=50, type=int)
    parser.add_argument("--anneal", default=False, type=bool)

    parser.add_argument("--train_batch", default=512, type=int)
    parser.add_argument("--val_batch", default=512, type=int)

    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--enable_local_test", default=False, type=bool)
    parser.add_argument("--test_size", default=0.3, type=float)

    parser.add_argument("--device", default="cpu")

    parser.add_argument("--n_epochs", default=50, type=int)

    parser.add_argument("--features", help="paths of features", nargs="*")
    parser.add_argument("--metadata", help="metadata to retrieve answer")

    args = parser.parse_args()

    logger = get_logger("lstm-gru-attention", "lstm-gru-attention")
    logger.info(
        f"hidden_size: {args.hidden_size}, linear_size: {args.linear_size}")
    logger.info(f"n_attention: {args.n_attention}, anneal: {args.anneal}")
    logger.info(
        f"train_batch: {args.train_batch}, val_batch: {args.val_batch}")
    logger.info(f"n_splits: {args.n_splits}, seed: {args.seed}")
    logger.info(f"enable_local_test: {args.enable_local_test}")
    logger.info(f"test_size: {args.test_size}")
    logger.info(f"n_epochs: {args.n_epochs}")
    logger.info(f"features: {args.features}")

    features = []

    for path in args.features:
        path = Path(path)
        assert path.exists()
        with open(path, "rb") as f:
            feats = pickle.load(f)
        feats = np.concatenate(feats)
        features.append(feats)
    train = np.concatenate(features, axis=2)

    answer = pd.read_csv(args.metadata).query("phase == 0").target.values

    trainer = NNTrainer(
        LSTMGRUAttentionNet,
        logger,
        n_splits=args.n_splits,
        seed=args.seed,
        enable_local_test=args.enable_local_test,
        test_size=args.test_size,
        device=args.device,
        train_batch=args.train_batch,
        val_batch=args.val_batch,
        kwargs={
            "hidden_size": args.hidden_size,
            "linear_size": args.linear_size,
            "input_shape": train.shape,
            "n_attention": args.n_attention
        },
        anneal=args.anneal)

    trainer.fit(train, answer, args.n_epochs)
    if args.enable_local_test:
        trainer.score()
