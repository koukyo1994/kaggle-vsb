import sys
import pickle

import torch
import torch.utils.data

import numpy as np
import pandas as pd

from pathlib import Path

from argparse import ArgumentParser

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("..")
    sys.path.append("./")
    from script.common.utils import get_logger
    from script.common.train_helpers import sigmoid
    from model import LSTMAttentionNet

    parser = ArgumentParser()
    parser.add_argument("--tag")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--features", help="path of features", nargs="*")
    parser.add_argument("--sample", default="../input/sample_submission.csv")

    args = parser.parse_args()
    logger = get_logger("lstm-gru-attention-test", "lstm-gru-attention-test")
    logger.info(f"tag: {args.tag}")
    logger.info(f"device: {args.device}")
    logger.info(f"features: {args.features}")

    features = []
    batch_size = 512

    for path in args.features:
        path = Path(path)
        assert path.exists()
        with open(path, "rb") as f:
            feats = pickle.load(f)
        if isinstance(feats, list):
            feats = np.concatenate(feats)
        features.append(feats)
    test = np.concatenate(features, axis=2)
    test_tensor = torch.tensor(test, dtype=torch.float32).to(args.device)
    dataset = torch.utils.data.TensorDataset(test_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    with open(f"trainer/{args.tag}/trainer.pkl", "rb") as f:
        trainer = pickle.load(f)

    bin_path = Path(f"bin/{args.tag}")
    test_preds = np.zeros(test.shape[0])
    for path in bin_path.iterdir():
        model = LSTMAttentionNet(**trainer.kwargs)
        model.to(args.device)
        model.load_state_dict(torch.load(path))

        model.eval()
        temp = np.zeros(test.shape[0])
        for i, (x_batch, ) in enumerate(loader):
            with torch.no_grad():
                y_pred = model(x_batch).detach()
                temp[i * batch_size:(i + 1) * batch_size] = sigmoid(
                    y_pred.cpu().numpy())[:, 0]
        test_preds += temp / trainer.n_splits
    prob_path = Path(f"probability/{args.tag}")
    prob_path.mkdir(exist_ok=True, parents=True)
    with open(prob_path / "probability.pkl", "wb") as f:
        pickle.dump(test_preds, f)

    submission_path = Path("submission")
    submission_path.mkdir(exist_ok=True)
    sub = pd.read_csv(args.sample)
    sub["target"] = (test_preds > trainer.threshold).astype(int)
    sub.to_csv(submission_path / f"{trainer.tag}.csv", index=False)
