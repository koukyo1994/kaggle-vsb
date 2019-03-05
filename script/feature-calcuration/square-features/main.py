import sys
import pickle

from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("..")
    sys.path.append("./")

    from script.common.utils import get_logger
    from feature_extraction import square_data

    parser = ArgumentParser()
    parser.add_argument(
        "--path", default="../features/basic-features/160d/train_basic.pkl")
    parser.add_argument("--scaler", default="")
    parser.add_argument("--name", default="square_train.pkl")

    args = parser.parse_args()
    path = Path(args.path)

    rel_path = path.parent.relative_to("../features")
    outdir = Path("../features/square-features") / rel_path
    outdir.mkdir(exist_ok=True, parents=True)

    logger = get_logger(
        name="square", tag=f"{str('square_features' / rel_path)}")
    logger.info(f"path: {args.path}, name: {args.name}")

    if args.scaler == "":
        scaler = None
    else:
        scaler_path = Path(args.scaler)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    scaler, features = square_data(path, scaler)
    logger.info(f"X_shape: {features.shape}")
    with open(outdir / args.name, "wb") as f:
        pickle.dump(features, f)

    with open(outdir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
