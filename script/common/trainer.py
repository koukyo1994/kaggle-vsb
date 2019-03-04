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
    def __init__(self,
                 logger,
                 n_splits=5,
                 seed=42,
                 enable_local_test=False,
                 test_size=0.3):
        self.enable_local_test = enable_local_test
        self.test_size = test_size
        self.n_splits = n_splits
        self.seed = seed
        self.logger = logger

        self.fold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        self.best_score = None
        self.best_threshold = None
        self.tag = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

    def _split(self, train, answer):
        X_train, local_test, y_train, y_local_test = train_test_split(
            train, answer, test_size=self.test_size, random_state=self.seed)
        self.train_set = X_train
        self.y = y_train
        self.local_test_set = local_test
        self.y_local = y_local_test
        self.logger.info(
            f"local_test_size: {self.test_size}, random_state: {self.seed}")

    def fit(self, train, answer, n_epochs=50):
        if self.enable_local_test:
            self._split(train, answer)
        else:
            self.train_set = train
            self.y = answer
            self.local_test_set = None
            self.y_local = None

        self.train_preds = np.zeros((self.train_set.shape[0]))
        for i, (trn_index, val_index) in enumerate(
                self.fold.split(self.train_set, self.y)):
            self.fold = i
            self.logger.info(f"\nFold {i+1}")
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
        self.best_score = search_result["mcc"]
        self.best_threshold = search_result["threshold"]

    def score(self):
        self._score(self.local_test_set, self.y_local)


class NNTrainer(Trainer):
    def __init__(self,
                 model,
                 logger,
                 n_splits=5,
                 seed=42,
                 enable_local_test=False,
                 test_size=0.3,
                 device="cpu",
                 train_batch=128,
                 val_batch=512,
                 kwargs={},
                 anneal=True):
        super(NNTrainer, self).__init__(
            logger=logger,
            n_splits=n_splits,
            seed=seed,
            enable_local_test=enable_local_test,
            test_size=test_size)
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.anneal = anneal
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean").to(self.device)

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
        if len(eval_set) == 2:
            x_val = torch.tensor(
                eval_set[0], dtype=torch.float32).to(self.device)
            y_val = torch.tensor(
                eval_set[1][:, np.newaxis],
                dtype=torch.float32).to(self.device)
            valid = torch.utils.data.TensorDataset(x_val, y_val)
            valid_loader = torch.utils.data.DataLoader(
                valid, batch_size=self.val_batch, shuffle=False)
        model = self.model(**self.kwargs)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters())
        if self.anneal:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs)
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
                val_mcc, val_threshold = search_result["mcc"], search_result[
                    "threshold"]
            self.logger.info(
                f"loss: {avg_loss:.4f} val_loss: {avg_val_loss:.4f}")
            self.logger.info(f"val_mcc: {val_mcc} best_t: {val_threshold}")
            if self.anneal:
                scheduler.step()
            if val_mcc > best_score:
                torch.save(model.state_dict(),
                           self.path / f"best{self.fold}.pt")
                self.logger.info(f"Save model on epoch {epoch+1}")
                best_score = val_mcc
        model.load_state_dict(torch.load(self.path / f"best{self.fold}.pt"))
        valid_preds, avg_val_loss = self._val(valid_loader, model)
        self.logger.info(f"Validation loss: {avg_val_loss}")
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
                valid_preds[i * self.val_batch:(i + 1) *
                            self.val_batch] = sigmoid(
                                y_pred.cpu().numpy())[:, 0]
        return valid_preds, avg_val_loss

    def _score(self, X, y):
        model = self.model(**self.kwargs)
        model.to(self.device)

        x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(
            y[:, np.newaxis], dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        batch_size = 128
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False)

        test_preds = np.zeros(X.shape[0])
        for path in self.path.iterdir():
            model.load_state_dict(torch.load(path))
            model.eval()
            temp = np.zeros(X.shape[0])
            for i, (x_batch, y_batch) in enumerate(loader):
                with torch.no_grad():
                    y_pred = model(x_batch).detach()
                    temp[i * batch_size:(i + 1) * batch_size] = sigmoid(
                        y_pred.cpu().numpy())[:, 0]
            test_preds += temp / self.n_splits
        search_result = threshold_search(y, test_preds)
        self.logger.info(f"local test MCC: { search_result['mcc']}")
        self.logger.info(f"local test threshold: {search_result['threshold']}")
        self.best_score = search_result["mcc"]
        self.best_threshold = search_result["threshold"]
