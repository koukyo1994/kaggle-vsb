import sys
import pickle
import numpy as np

from pathlib import Path
from argparse import ArgumentParser

from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")

    from script.common.utils import get_logger

    parser = ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--train_name", default="train.pkl")
    parser.add_argument("--test_name", default="test.pkl")

    args = parser.parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)

    rel_path = train_path.parent.relative_to("../features")
    outdir = Path("../features/scaled-features") / rel_path
    outdir.mkdir(exist_ok=True, parents=True)

    logger = get_logger(
        name="scaled", tag=f"(str('scaled_features') / rel_path)")
    logger.info(f"train: {args.train}, test: {args.test}")
    logger.info(f"trian_name: {args.train_name}, test_name: {args.test_name}")

    with open(args.train, "rb") as f:
        train = pickle.load(f)

    with open(args.test, "rb") as f:
        test = pickle.load(f)

    X_all = np.concatenate([train, test])
    scalers = {}
    for row in range(X_all.shape[1]):
        scalers[row] = StandardScaler()
        scalers[row].fit(X_all[:, row, :])

    for row in range(train.shape[1]):
        train[:, row, :] = scalers[row].transform(train[:, row, :])

    for row in range(test.shape[1]):
        test[:, row, :] = scalers[row].transform(test[:, row, :])

    with open(outdir / args.train_name, "wb") as f:
        pickle.dump(train, f)

    with open(outdir / args.test_name, "wb") as f:
        pickle.dump(test, f)

    with open(outdir / "scaler.pkl", "wb") as f:
        pickle.dump(scalers, f)
