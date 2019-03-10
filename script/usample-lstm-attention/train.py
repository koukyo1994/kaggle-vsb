import sys
import pickle
import numpy as np
import pandas as pd

from pathlib import Path

from argparse import ArgumentParser

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("..")
    sys.path.append("./")
    from model import LSTMAttentionNet
    from script.common.trainer import NNTrainer
    from script.common.utils import get_logger

    parser = ArgumentParser()
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--linear_size", default=100, type=int)
    parser.add_argument("--n_attention", default=50, type=int)
    parser.add_argument("--anneal", action="store_true")

    parser.add_argument("--train_batch", default=512, type=int)
    parser.add_argument("--val_batch", default=512, type=int)

    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--enable_local_test", action="store_true")
    parser.add_argument("--test_size", default=0.3, type=float)
    parser.add_argument("--scaling", action="store_true")

    parser.add_argument("--device", default="cpu")

    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument("--sample_ratio", default=10.0, type=float)

    parser.add_argument("--features", help="paths of features", nargs="*")
    parser.add_argument("--metadata", help="metadata to retrieve answer")

    args = parser.parse_args()

    logger = get_logger("lstm-attention", "lstm-attention")
    logger.info(
        f"hidden_size: {args.hidden_size}, linear_size: {args.linear_size}")
    logger.info(f"n_attention: {args.n_attention}, anneal: {args.anneal}")
    logger.info(
        f"train_batch: {args.train_batch}, val_batch: {args.val_batch}")
    logger.info(f"n_splits: {args.n_splits}, seed: {args.seed}")
    logger.info(f"enable_local_test: {args.enable_local_test}")
    logger.info(f"test_size: {args.test_size}")
    logger.info(f"n_epochs: {args.n_epochs}")
    logger.info(f"Under sampling ratio: {args.sample_ratio}")
    logger.info(f"features: {args.features}")

    features = []

    for path in args.features:
        path = Path(path)
        assert path.exists()
        with open(path, "rb") as f:
            feats = pickle.load(f)
        if isinstance(feats, list):
            feats = np.concatenate(feats)
        features.append(feats)
    train = np.concatenate(features, axis=2)

    scaler = {}
    if args.scaling:
        logger.info("scaling...")
        for i in range(train.shape[1]):
            scaler[i] = StandardScaler()
            train[:, i, :] = scaler[i].fit_transform(train[:, i, :])

    answer = pd.read_csv(args.metadata).query("phase == 0").target.values

    n_true = len(answer[answer == 1])
    idx = np.arange(0, len(answer)).reshape(-1, 1)
    rus = RandomUnderSampler(
        sampling_strategy={
            1: n_true,
            0: int(args.sample_ratio * n_true)
        },
        random_state=args.seed)
    use_idx, y = rus.fit_resample(idx, answer)
    X = train[use_idx.reshape(-1)]

    trainer = NNTrainer(
        LSTMAttentionNet,
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

    trainer.fit(X, y, args.n_epochs)
    if args.enable_local_test:
        trainer.score()

    trainer_path = Path(f"trainer/{trainer.tag}")
    trainer_path.mkdir(parents=True, exist_ok=True)

    trainer.fold = None
    trainer.local_test_set = None
    trainer.train_set = None
    trainer.y = None
    trainer.y_local = None
    trainer.logger = None
    trainer.model = None
    with open(trainer_path / "trainer.pkl", "wb") as f:
        pickle.dump(trainer, f)

    with open(trainer_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
