from typing import List, Optional

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from constants import *
from transforms import *


class PreprocessDataLoader(DataLoader):
    def __iter__(self):
        batches = super().__iter__()
        for b in batches:
            yield self.preprocess(*b)

    def preprocess(self, x, y):
        return x, y


class GPUDataLoader(PreprocessDataLoader):
    def preprocess(self, x, y):
        return x.float().to(DEVICE), y.to(DEVICE)


def load_file(path: str, is_labels=False, augmentations: List[ReversibleTransform] = [Identity()]):
    assert augmentations

    input = torch.tensor(np.load(path), dtype=torch.long if is_labels else torch.float)
    input = input.permute((0, 3, 1, 2))
    input = input // 255 if is_labels else input / 255
    input = input.squeeze()

    tensors = [aug.apply(input) for aug in augmentations]
    input = torch.cat(tensors, dim=0)

    return input


def get_dataloader(
        x_path: str,
        y_path: str,
        shuffle: bool,
        drop_last: bool,
        mb_size=DEFAULT_MB_SIZE,
        augmentations = [Identity()],
):
    xs = load_file(x_path, augmentations=augmentations)
    ys = load_file(y_path, augmentations=augmentations, is_labels=True)

    ds = TensorDataset(xs, ys)
    return GPUDataLoader(dataset=ds, batch_size=mb_size, shuffle=shuffle, drop_last=drop_last, pin_memory=True)


def get_dataloaders(train_augmentations=[Identity()]):
    train_dl = get_dataloader(TRAIN_X_PATH, TRAIN_Y_PATH, shuffle=True, drop_last=True, augmentations=train_augmentations)
    test_dl = get_dataloader(TEST_X_PATH, TEST_Y_PATH, shuffle=False, drop_last=False)
    valid_dl = test_dl

    return train_dl, valid_dl, test_dl
