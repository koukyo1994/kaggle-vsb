from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedKFold

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

            valid_preds, local_preds = self._fit(
                X_train, y_train, n_epochs, eval_set=(X_val, y_val))
            self.train_preds[val_index] = valid_preds
        search_result = threshold_search(self.y, self.train_preds)
        search_result_loc = threshold_search(self.valid_set[1], local_preds)

        self.logger.info(f"MCC: {search_result['mcc']}")
        self.logger.info(f"threshold: {search_result['threshold']}")
        self.logger.info(f"Local MCC: {search_result_loc['mcc']}")
        self.logger.info(f"Local threshold: {search_result_loc['threshold']}")

        self.best_score = search_result['mcc']
        self.best_threshold = search_result['threshold']
        self.av_score = search_result_loc['mcc']
        self.av_threshold = search_result_loc['threshold']


class NNTrainer(Trainer):
    def __init__(self,
                 model,
                 logger,
                 av_valid,
                 n_splits=5,
                 seed=42,
                 device="cpu",
                 train_batch=128,
                 kwargs={}):
        super().__init__(logger, av_valid, n_splits=n_splits, seed=seed)
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.train_batch = train_batch
        self.val_batch = 128

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean").to(self.device)

        local_test = torch.tensor(
            self.valid_set[0], dtype=torch.float32).to(self.device)
        local_test_y = torch.tensor(
            self.valid_set[1][:, np.newaxis],
            dtype=torch.float32).to(self.device)
        local_set = torch.utils.data.TensorDataset(local_test, local_test_y)
        self.local_loader = torch.utils.data.DataLoader(
            local_set, batch_size=128, shuffle=False)

        path = Path(f"bin/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path

    def _fit(self, X_train, y_train, n_epochs=50, eval_set=()):
        seed_torch()
        x_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(
            y_train[:, np.newaxis], dtype=torch.float32).to(self.device)

        train = torch.utils.data.TensorDataset(x_train, y)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=self.train_batch, shuffle=True)

        x_val = torch.tensor(eval_set[0], dtype=torch.float32).to(self.device)
        y_val = torch.tensor(
            eval_set[1][:, np.newaxis], dtype=torch.float32).to(self.device)
        valid = torch.utils.data.TensorDataset(x_val, y_val)
        valid_loader = torch.utils.data.DataLoader(
            valid, batch_size=self.val_batch, shuffle=False)

        model = self.model(**self.kwargs)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters())
        best_score = -np.inf
        best_loc_score = -np.inf

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
                val_mcc, val_threshold = search_result["mcc"], search_result[
                    "threshold"]

                local_preds, avg_loc_loss = self._val(self.local_loader, model)
                search_result_loc = threshold_search(self.valid_set[1],
                                                     local_preds)
                loc_mcc, loc_threshold = search_result_loc[
                    "mcc"], search_result_loc["threshold"]
            self.logger.info(
                f"loss: {avg_loss:.4f} val_loss: {avg_val_loss:.4f}")
            self.logger.info(f"val_mcc: {val_mcc:.4f} best_t: {val_threshold}")
            self.logger.info(f"loc_mcc: {loc_mcc:.4f} loc_t: {loc_threshold}")
            if val_mcc > best_score and loc_mcc > best_loc_score:
                torch.save(model.state_dict(),
                           self.path / f"best{self.fold}.pt")
                self.logger.info(f"Save model on epoch {epoch+1}")
                best_score = val_mcc
                best_loc_score = loc_mcc
            elif val_mcc > best_score:
                best_score = val_mcc
            elif loc_mcc > best_loc_score:
                best_loc_score = loc_mcc
        model.load_state_dict(torch.load(self.path / f"best{self.fold}.pt"))
        valid_preds, avg_val_loss = self._val(valid_loader, model)
        local_preds, avg_loc_loss = self._val(self.local_loader, model)
        self.logger.info(f"Validation loss: {avg_val_loss:.4f}")
        self.logger.info(f"local loss: {avg_loc_loss:.4f}\n")
        return valid_preds, local_preds

    def _val(self, loader, model):
        model.eval()
        valid_preds = np.zeros(loader.dataset.tensors[0].size(0))
        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(loader):
            with torch.no_grad():
                y_pred = model(x_batch).detach()
                avg_val_loss += self.loss_fn(y_pred,
                                             y_batch).item() / len(loader)
                valid_preds[i * self.val_batch:(i + 1) *
                            self.val_batch] = sigmoid(
                                y_pred.cpu().numpy())[:, 0]
        return valid_preds, avg_val_loss
