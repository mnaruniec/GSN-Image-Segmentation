import matplotlib.pyplot as plt
import torch

from constants import *


def show_censored_img(x, y):
    img = x.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i, j, k] = img[i, j, k] if y[i, j] else 0

    plt.imshow(img)
    plt.show()


def iou(target, pred, verify_size=True):
    if verify_size:
        assert target.shape == pred.shape
        assert target.shape == (IMG_SIZE, IMG_SIZE)

    inter = torch.logical_and(target, pred)
    union = torch.logical_or(target, pred)

    return torch.sum(inter).float() / torch.sum(union)


def entropy(x):
    x = torch.nn.Softmax2d()(x)
    log = torch.log2(x)
    x = -x * log
    return torch.sum(x, dim=-3)