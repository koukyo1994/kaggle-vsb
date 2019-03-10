import sys
import pickle

import numpy as np

import torch
import torch.utils.data

from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")
    from model import LSTMAttentionNet
    from script.common.train_helpers import sigmoid, threshold_search

    parser = ArgumentParser()
    parser.add_argument("--tag")
    parser.add_argument("--validation_set")

    args = parser.parse_args()

    trainer_dir = Path(f"trainer/{args.tag}/trainer.pkl")
    with open(trainer_dir, "rb") as f:
        trainer = pickle.load(f)

    with open(args.validation_set, "rb") as f:
        valid = pickle.load(f)

    x = torch.tensor(valid[0], dtype=torch.float32).to("cpu")
    y = torch.tensor(valid[1][:, np.newaxis], dtype=torch.float32).to("cpu")
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False)

    bin_dir = Path(f"bin/{args.tag}")
    model = LSTMAttentionNet(**trainer.kwargs)
    model.to("cpu")
    preds = np.zeros(valid[0].shape[0])
    for path in bin_dir.iterdir():
        model.load_state_dict(torch.load(path))
        model.eval()
        temp = np.zeros_like(preds)
        for i, (x_batch, y_batch) in enumerate(loader):
            y_pred = model(x_batch).detach()
            temp[i * 128:(i + 1) * 128] = sigmoid(y_pred.cpu().numpy())[:, 0]
        preds += temp / 5
    search_result = threshold_search(valid[1], preds)
    print(f"MCC: {search_result['mcc']:.4f}")
    print(f"threshold: {search_result['threshold']}")
