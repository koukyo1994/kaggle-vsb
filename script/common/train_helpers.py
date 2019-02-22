import os
import random
import torch
import numpy as np

from sklearn.metrics import matthews_corrcoef


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:
        score = matthews_corrcoef(y_true, y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {"threshold": best_threshold, "mcc": best_score}
    return search_result


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
