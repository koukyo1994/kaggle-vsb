import sys
import pickle
import numpy as np

from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")
    from model import LSTMAttentionNet
    from script.common.utils import get_logger
    from trainer import NNTrainer

    parser = ArgumentParser()
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--linear_size", default=100, type=int)
    parser.add_argument("--n_attention", default=50, type=int)

    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--n_trial", default=10, type=int)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_epochs", default=30, type=int)

    parser.add_argument("--allow", default=0.3, type=float)

    parser.add_argument(
        "--train", default="../features/basic-features/160d/train_basic.pkl")
    parser.add_argument(
        "--test", default="../features/basic-features/160d/basic_test.pkl")
    args = parser.parse_args()

    data_path = Path(args.train)
    rel_path = data_path.parent.relative_to("../features")
    logger = get_logger("adversarial-validation", str(rel_path))
    logger.info(
        f"hidden_size: {args.hidden_size}, linear_size: {args.linear_size}")
    logger.info(f"n_attention: {args.n_attention}")
    logger.info(f"n_splits: {args.n_splits}, n_trial: {args.n_trial}")
    logger.info(f"device: {args.device}, n_epochs: {args.n_epochs}")
    logger.info(f"allow: {args.allow}")
    logger.info(f"train: {args.train}")
    logger.info(f"test: {args.test}")

    with open(Path(args.train), "rb") as f:
        train = pickle.load(f)

    with open(Path(args.test), "rb") as f:
        test = pickle.load(f)

    n_train = train.shape[0]
    n_test = test.shape[0]
    n_all = n_train + n_test

    y_train = np.ones((n_train, 1))
    y_test = np.zeros((n_test, 1))
    y = np.vstack([y_train, y_test]).reshape(-1)

    X = np.vstack([train, test])

    trainer = NNTrainer(
        LSTMAttentionNet,
        logger,
        n_splits=args.n_splits,
        n_trial=args.n_trial,
        device=args.device,
        kwargs={
            "hidden_size": args.hidden_size,
            "linear_size": args.linear_size,
            "input_shape": X.shape,
            "n_attention": args.n_attention
        },
        name=str(rel_path))
    trainer.fit(X, y, args.n_epochs)
    train_mask, test_mask, train_result, test_result = trainer.av_results(
        args.allow)

    mask_dir = Path("mask" / rel_path)
    mask_dir.mkdir(exist_ok=True, parents=True)
    with open(mask_dir / "train_mask.pkl", "wb") as f:
        pickle.dump(train_mask, f)

    with open(mask_dir / "test_mask.pkl", "wb") as f:
        pickle.dump(test_mask, f)

    with open(mask_dir / "train_result.pkl") as f:
        pickle.dump(train_result, f)

    with open(mask_dir / "test_result.pkl", "wb") as f:
        pickle.dump(test_result, f)
