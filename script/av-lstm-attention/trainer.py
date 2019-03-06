from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split

from script.common.utils import timer
from script.common.train_helpers import (sigmoid, threshold_search, seed_torch)


class Trainer:
    def __init__(self, logger, av_valid, n_splits=5, seed=42):
        self.logger = logger
        self.valid_set = av_valid
        self.n_splits = n_splits
        self.seed = seed

        self.fold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        self.best_score = None
        self.best_threshold = None

        self.av_score = None
        self.av_threshold = None
        self.tag = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

    def fit(self, train, answer, n_epochs=50):
        self.train_set = train
        self.y = answer

        self.train_preds = np.zeros((self.train_set.shape[0]))
        for i, (trn_index, val_index) in enumerate(
                self.fold.split(self.train_set, self.y)):
            self.fold = i
            self.logger.info(f"Fold {i+1}")
            X_train = self.train_set[trn_index]
            X_val = self.train_set[val_index]
            y_train = self.y[trn_index]
            y_val = self.y[val_index]

            valid_preds = self._fit(
                X_train, y_train, n_epochs, eval_set=(X_val, y_val))
            self.train_preds[val_index] = valid_preds
        search_result = threshold_search(self.y, self.train_preds)

        self.logger.info(f"MCC: {search_result['mcc']}")
        self.logger.info(f"threshold: {search_result['threshold']}")
        self.best_score = search_result['mcc']
        self.best_threshold = search_result['threshold']

        self._score(self.valid_set, self.valid_y)


class NNTrainer(Trainer):
    def __init__(self,
                 model,
                 logger,
                 av_valid,
                 n_splits=5,
                 seed=42,
                 device="cpu",
                 train_batch=128,
                 val_batch=512,
                 kwargs={}):
        super().__init__(logger, av_valid, n_splits=n_splits, seed=seed)
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.train_batch = train_batch
        self.val_batch = val_batch

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean").to(self.device)

        path = Path(f"bin/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path

    def _fit(self, X_train, y_train, n_epochs=50, eval_set=()):
        seed_torch()
        x_train = torch.tensor(X_train, dtype=torch.float32)
