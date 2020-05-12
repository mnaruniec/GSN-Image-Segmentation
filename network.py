import os
import random

from copy import deepcopy
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt

from constants import *
from input import get_dataloaders


class SegNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.layers = [
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1)
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class SegTrainer:
    def __init__(
            self,
            optimizer_lambda = partial(optim.Adam, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY, amsgrad=True),
            mb_size=DEFAULT_MB_SIZE,
            num_epochs=DEFAULT_NUM_EPOCHS,
            patience=DEFAULT_PATIENCE,
            stat_period=DEFAULT_STAT_PERIOD,
            epoch_train_eval=DEFAULT_EPOCH_TRAIN_EVAL,
            **net_kwargs,
    ):
        self.net_kwargs = net_kwargs

        self.optimizer_lambda = optimizer_lambda

        self.mb_size = mb_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.stat_period = stat_period
        self.epoch_train_eval = epoch_train_eval

        self.net = None
        self.criterion = None
        self.optimizer = None

        self.train_dl, self.valid_dl, self.test_dl = get_dataloaders()

    def init_net(self):
        self.net = SegNet(**self.net_kwargs)
        self.net.to(DEVICE)
        self.net.train()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer_lambda(self.net.parameters())

    def evaluate_on(self, dataloader: DataLoader, full=False) -> (float, int, float):
        # TODO add IOU
        with torch.no_grad():
            net = self.net
            net.eval()

            correct = 0
            total = 0

            running_loss = 0.
            i = 0

            for data in dataloader:
                i += 1
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) * labels.size(1) * labels.size(2)
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                if not full and i >= self.stat_period:
                    break

        net.train()
        return correct / total, total, running_loss / i

    def run_evaluation(self, dataloader, ds_name: str):
        acc, total, loss = self.evaluate_on(dataloader, full=True)

        print(f'{ds_name} stats: acc: {(100 * acc):.2f}%, loss: {loss:.4f}')

        return acc, loss

    def train_batch(self, data):
        inputs, labels = data

        self.net.train()
        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, reset_net=True, plot_loss=True):
        if reset_net:
            self.init_net()

        train_losses = []
        valid_losses = []
        epoch_losses = []
        epoch_x = 0
        epoch_xs = []

        last_epoch = 0
        best_state_dict = None
        best_epoch = 0
        best_epoch_loss = 10 ** 9

        try:
            for epoch in range(1, self.num_epochs + 1):
                print(f'EPOCH {epoch}')

                train_loss = 0.0
                last_epoch = epoch

                for i, data in enumerate(self.train_dl, 0):
                    train_loss += self.train_batch(data)

                    if i % self.stat_period == self.stat_period - 1:
                        epoch_x += 1

                        train_loss = train_loss / self.stat_period
                        train_losses.append(train_loss)

                        acc, total, valid_loss = self.evaluate_on(self.valid_dl)

                        valid_losses.append(valid_loss)

                        print('Epoch %d, batch %d, train loss: %.4f, valid loss: %.4f' %
                              (epoch, i + 1, train_loss, valid_loss))

                        train_loss = 0.0

                _, epoch_loss = self.run_evaluation(self.valid_dl, 'VALID')

                epoch_losses.append(epoch_loss)
                epoch_xs.append(epoch_x)

                # early stopping & snapshotting
                if epoch_loss < best_epoch_loss:
                    best_epoch_loss = epoch_loss
                    best_epoch = epoch
                    best_state_dict = deepcopy(self.net.state_dict())
                elif len(epoch_losses) > self.patience:
                    if all((l > best_epoch_loss for l in epoch_losses[-(self.patience + 1):])):
                        print(f'No improvement in last {self.patience + 1} epochs, early stopping.')
                        break

                if self.epoch_train_eval:
                    self.run_evaluation(self.train_dl, 'TRAIN')

        finally:
            if plot_loss:
                plt.plot(
                    range(len(train_losses)), train_losses, 'r',
                    range(len(valid_losses)), valid_losses, 'b',
                    epoch_xs, epoch_losses, 'g',
                )
                plt.show()

            if best_state_dict and best_epoch != last_epoch:
                print(f'Restoring snapshot from epoch {best_epoch} with valid loss: {best_epoch_loss:.4f}')
                self.net.load_state_dict(best_state_dict)

            acc, loss = self.run_evaluation(self.test_dl, 'TEST')
            self.run_evaluation(self.train_dl, 'TRAIN')

            return acc, loss
