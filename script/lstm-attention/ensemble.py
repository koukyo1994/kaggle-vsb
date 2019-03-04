import sys
import pickle
import numpy as np
import pandas as pd

from pathlib import Path

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("../")
    path = Path("probability")
    pickles = []
    thresholds = []
    for p in path.iterdir():
        with open(p / "probability.pkl", "rb") as f:
            pic = pickle.load(f)
            pickles.append(pic)

    thres_path = Path("trainer")
    for t_path in thres_path.iter_dir():
        with open(t_path / "trainer.pkl", "rb") as f:
            tr = pickle.load(f)
            thresholds.append(tr.best_threshold)

    preds = np.zeros_like(pickles[0])
    n_preds = len(pickles)
    for p in pickles:
        preds += p / n_preds
    
    thres = np.array(thresholds).mean()
    
    pred_3 = []
    for p in preds:
        for i in range(3):
            pred_3.append(p)
    pred_3 = np.array(pred_3)
    submission_path = Path("submission")
    sub = pd.read_csv("../input/sample_submission.csv")
    sub.target = (pred_3 > thres).astype(int)
    sub.to_csv(submission_path / "ensemble.csv", index=False)
