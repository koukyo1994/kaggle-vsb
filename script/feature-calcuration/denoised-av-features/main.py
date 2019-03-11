import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--train", default="../features/denoised-basic/160d/dnhp/train.pkl")
    parser.add_argument("--av")
    args = parser.parse_args()

    feature_path = Path(args.train)
    av_path = Path(args.av)
    rel_path = av_path.parent.relative_to("../adversarial-validation/mask/")
    out_path = Path("../features/av-features") / rel_path
    out_path = out_path / f"th_1.0"
    out_path.mkdir(exist_ok=True, parents=True)

    with open(feature_path, "rb") as f:
        X_train = pickle.load(f)

    with open(av_path, "rb") as f:
        prob_mask = pickle.load(f)

    meta = pd.read_csv("../input/metadata_train.csv")

    mask = (prob_mask == 1.0)
    mask3 = []
    for row in mask:
        for _ in range(3):
            mask3.append(row)
    mask3 = np.array(mask3)

    target = meta.target.values

    train = X_train[~mask]
    target = target[~mask3][::3]

    assert len(train) == len(target)

    train_set = (train, target)

    with open(out_path, "train.pkl", "wb") as f:
        pickle.dump(train_set, f)
