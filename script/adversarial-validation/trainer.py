from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np

from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from script.common.utils import timer
from script.common.train_helpers import (sigmoid, seed_torch)


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true, y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {"threshold": best_threshold, "f1": best_score}
    return search_result


class Trainer:
    def __init__(self, logger, n_splits=5, n_trial=10):
        self.logger = logger
        self.n_splits = n_splits
        self.n_trial = n_trial

        self.folds = [
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
            for i in range(n_trial)
        ]
        self.best_scores = []
        self.best_thresholds = []
        self.tag = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.av_results_on_train = []
        self.av_results_on_test = []

    def fit_once(self, X, y, n_epochs=50, iter_number=0):
        self.train_preds = np.zeros((X.shape[0]))
        for i, (trn_index,
                val_index) in enumerate(self.folds[iter_number].split(X, y)):
            self.fold = i
            self.logger.info(f"Fold {i + 1} on iter_number: {iter_number}")
            X_train = X[trn_index]
            X_val = X[val_index]
            y_train = y[trn_index]
            y_val = y[val_index]

            valid_preds = self._fit(
                X_train, y_train, n_epochs, eval_set=(X_val, y_val))
            self.train_preds[val_index] = valid_preds
        search_result = threshold_search(y, self.train_preds)

        self.logger.info(f"Search Result for iter_number: {iter_number}")
        self.logger.info(f"F1: {search_result['f1']}")
        self.logger.info(f"threshold: {search_result['threshold']}")
        self.best_scores.append(search_result["f1"])
        self.best_thresholds.append(search_result["threshold"])

        self.av_results_on_train.append(
            (self.train_preds >
             search_result["threshold"]).astype(int)[y == 1])
        self.av_results_on_test.append(
            (self.train_preds >
             search_result["threshold"]).astype(int)[y == 0])

    def fit(self, X, y, n_epochs=50):
        for i in range(self.n_trial):
            self.fit_once(X, y, n_epochs=n_epochs, iter_number=i)

    def av_results(self, allow=0.3):
        train_result = np.mean(self.av_results_on_train, axis=0)
        test_result = np.mean(self.av_results_on_test, axis=0)

        train_mask = np.logical_and(train_result >= allow,
                                    train_result <= 1 - allow)
        test_mask = np.logical_and(test_result >= allow,
                                   test_result <= 1 - allow)
        return train_mask, test_mask, train_result, test_result


class NNTrainer(Trainer):
    def __init__(self,
                 model,
                 logger,
                 n_splits=5,
                 n_trial=10,
                 device="cpu",
                 kwargs={},
                 name="basic-160d"):
        super().__init__(logger, n_splits=n_splits, n_trial=n_trial)
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.name = name

        self.train_batch = 128
        self.valid_batch = 512

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean").to(self.device)
        path = Path(f"bin/{name}/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path

    def _fit(self, X_train, y_train, n_epochs=50, eval_set=()):
        seed_torch()
        x = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(
            y_train[:, np.newaxis], dtype=torch.float32).to(self.device)
        train = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=self.train_batch, shuffle=True)
        if len(eval_set) == 2:
            x_val = torch.tensor(
                eval_set[0], dtype=torch.float32).to(self.device)
            y_val = torch.tensor(
                eval_set[1][:, np.newaxis],
                dtype=torch.float32).to(self.device)
            valid = torch.utils.data.TensorDataset(x_val, y_val)
            valid_loader = torch.utils.data.DataLoader(
                valid, batch_size=self.valid_batch, shuffle=False)
        model = self.model(**self.kwargs)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters())
        best_score = -np.inf

        for epoch in range(n_epochs):
            with timer(f"Epoch {epoch+1}/{n_epochs}", self.logger):
                model.train()
                avg_loss = 0.
                for (x_batch, y_batch) in train_loader:
                    y_pred = model(x_batch)
                    loss = self.loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(train_loader)
                valid_preds, avg_val_loss = self._val(valid_loader, model)
                search_result = threshold_search(eval_set[1], valid_preds)
                val_f1, val_threshold = search_result["f1"], search_result[
                    "threshold"]
            self.logger.info(
                f"loss: {avg_loss:.4f} val_loss: {avg_val_loss:.4f}")
            self.logger.info(f"val_f1: {val_f1:.4f} best_t: {val_threshold}\n")

            if val_f1 > best_score:
                torch.save(model.state_dict(),
                           self.path / f"best{self.fold}.pt")
                self.logger.info(f"Save model on epoch {epoch+1}")
                best_score = val_f1
        model.load_state_dict(torch.load(self.path / f"best{self.fold}.pt"))
        valid_preds, avg_val_loss = self._val(valid_loader, model)
        self.logger.info(f"Validation loss: {avg_val_loss:.4f}")
        return valid_preds

    def _val(self, loader, model):
        model.eval()
        valid_preds = np.zeros(loader.dataset.tensors[0].size(0))
        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(loader):
            with torch.no_grad():
                y_pred = model(x_batch).detach()
                avg_val_loss += self.loss_fn(y_pred,
                                             y_batch).item() / len(loader)
                valid_preds[i * self.valid_batch:(i + 1) *
                            self.valid_batch] = sigmoid(
                                y_pred.cpu().numpy())[:, 0]
        return valid_preds, avg_val_loss
