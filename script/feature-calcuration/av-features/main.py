import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train", default="../features/basic-features/160d/train_basic.pkl")
    parser.add_argument("--av")
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    feature_path = Path(args.train)
    rel_path = feature_path.parent.relative_to("../features")

    out_path = Path("../features/av-features") / rel_path
    out_path.mkdir(exist_ok=True, parents=True)

    with open(feature_path, "rb") as f:
        X_train = pickle.load(f)

    with open(Path(args.av), "rb") as f:
        prob_mask = pickle.load(f)

    meta = pd.read_csv("../input/metadata_train.csv")

    mask = (prob_mask < args.threshold)
    mask3 = []
    for row in mask:
        for _ in range(3):
            mask3.append(row)
    mask3 = np.array(mask3)

    target = meta.target.values

    validation_X = X_train[mask]
    train_X = X_train[~mask]

    validation_target = target[mask3][::3]
    train_target = target[~mask3][::3]

    assert len(validation_X) == len(validation_target)
    assert len(train_X) == len(train_target)

    validation_set = (validation_X, validation_target)
    train_set = (train_X, train_target)

    with open(out_path / "validation.pkl", "wb") as f:
        pickle.dump(validation_set, f)

    with open(out_path / "train.pkl", "wb") as f:
        pickle.dump(train_set, f)
