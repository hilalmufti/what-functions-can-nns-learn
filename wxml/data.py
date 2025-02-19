from toolz.sandbox.core import unzip

import torch
from torch.utils.data import DataLoader

def lunzip(xs):
    xs, ys = unzip(xs)
    return list(xs), list(ys)

def split(xs, p):
    n = len(xs)
    return xs[:int(n * p)], xs[int(n * p):]

def make_loader(xs, ys, batch_size):
    print(xs, ys)
    xs, ys = torch.tensor(xs, dtype=torch.float32, requires_grad=True), torch.tensor(ys, dtype=torch.float32, requires_grad=True)
    data = list(zip(xs, ys))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return loader


def make_splits(xs, ys):
    pairs = list(zip(xs, ys))
    train, test = split(pairs, 0.8)
    val, test = split(test, 0.5)
    return train, val, test