import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from pathlib import Path

from datetime import datetime as dt


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_shape, device, layer_dim={}):
        super(VariationalAutoEncoder, self).__init__()
        self.maxlen = input_shape[1]
        self.input_dim = input_shape[2]
        self.device = device
        self.layer_dim = layer_dim

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.input_dim, layer_dim["first"])
        self.fc_mu = nn.Linear(layer_dim["first"], layer_dim["middle"])
        self.fc_var = nn.Linear(layer_dim["first"], layer_dim["middle"])

        self.fc2 = nn.Linear(layer_dim["middle"], layer_dim["second"])
        self.fc3 = nn.Linear(layer_dim["second"], self.input_dim)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_var(h)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(
                std.size()).normal_()).to(self.device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = self.relu(self.fc2(z))
        return self.sigmoid(self.fc3(h))

    def forward_one(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward(self, x):
        out_tensor = torch.zeros_like(x).to(self.device)
        mu_tensor = torch.zeros((x.shape[0], x.shape[1],
                                 self.layer_dim["middle"]))
        var_tensor = torch.zeros((x.shape[0], x.shape[1],
                                  self.layer_dim["middle"]))
        for i in range(self.maxlen):
            out, mu, var = self.forward_one(x[:, i, :])
            mu_tensor[:, i, :] = mu
            var_tensor[:, i, :] = var
            out_tensor[:, i, :] = out
        return out_tensor, mu_tensor, var_tensor


def loss_fn(recon_x, x, mu, log_var):
    recon = F.mse_loss(recon_x, x)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kld


class VAETrainer:
    def __init__(self, model, logger, device="cpu", train_batch=128,
                 kwargs={}):
        self.model = model
        self.logger = logger
        self.device = device
        self.train_batch = train_batch
        self.kwargs = kwargs
        self.loss_fn = loss_fn
        self.tag = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = Path(f"vae-features/bin/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path

    def fit(self, X, X_test, n_epochs, n_fold):
        model = self.model(X.shape, self.device, layer_dim=self.kwargs)
        model = model.to(self.device)

        x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = data.TensorDataset(x_tensor)
        loader = data.DataLoader(
            dataset, batch_size=self.train_batch, shuffle=True)

        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        test_dataset = data.TensorDataset(test_tensor)
        test_loader = data.DataLoader(
            test_dataset, batch_size=128, shuffle=False)

        optimizer = optim.Adam(model.parameters())

        for epoch in range(n_epochs):
            model.train()
            avg_loss = 0.
            avg_val_loss = 0.
            for i, (x_batch, ) in enumerate(loader):
                recon_batch, mu, var = model(x_batch)
                optimizer.zero_grad()

                loss = self.loss_fn(recon_batch, x_batch, mu, var)
                loss.backward()
                avg_loss += loss.item() / len(loader)
                optimizer.step()
            self.logger.info(f"Train loss: {avg_loss:.4f} at epoch {epoch+1}")
            model.eval()
            for i, (x_batch, ) in enumerate(test_loader):
                with torch.no_grad():
                    recon_batch_test, mu_test, var_test = model(x_batch)
                    loss = self.loss_fn(recon_batch_test, x_batch, mu_test,
                                        var_test)
                    avg_val_loss += loss.item() / len(test_loader)
            self.logger.info(
                f"Test loss: {avg_val_loss:.4f} at epoch {epoch + 1}")
        torch.save(model.state_dict(), self.path / f"weight{n_fold}.pt")

    def predict(self, X, n_fold):
        model = self.model(X.shape, self.device, layer_dim=self.kwargs)
        model.load_state_dict(torch.load(self.path / f"weight{n_fold}.pt"))
        model = model.to(self.device)

        x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = data.TensorDataset(x_tensor)
        loader = data.DataLoader(
            dataset, batch_size=self.train_batch, shuffle=False)

        model.eval()
        n_middle = self.kwargs["middle"]
        X_pred = np.zeros((X.shape[0], X.shape[1], n_middle * 2 + 1))
        for i, (x_batch, ) in enumerate(loader):
            with torch.no_grad():
                out, mu, var = model(x_batch)
                for j in range(out.shape[1]):
                    recon = ((out[:, j, :] - x_batch[:, j, :])**2).mean(dim=1)
                    X_pred[i * self.train_batch:(i + 1) *
                           self.train_batch, j, 0] = recon

                X_pred[i * self.train_batch:(i + 1) *
                       self.train_batch, :, 1:n_middle +
                       1] = mu.detach().cpu().numpy()
                X_pred[i * self.train_batch:(i + 1) *
                       self.train_batch, :, n_middle +
                       1:] = var.detach().cpu().numpy()
        return X_pred
