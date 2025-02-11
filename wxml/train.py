import torch
import tqdm


def accuracy(x, y):
    acc = torch.sum(x == y) / len(x)
    return acc.item()


def train_step(model, loss_fn, opt, x, y):
    assert model.training and x.requires_grad and y.requires_grad
    opt.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    opt.step()
    return y_pred, loss.item()


def train(model, loss_fn, opt, loader, epochs=10):
    for epoch in (pbar := tqdm.trange(epochs, desc="loss: inf, acc: inf")):
        for x, y in loader:
            y_pred, loss = train_step(model, loss_fn, opt, x, y)
            acy = accuracy(y_pred, y)
            pbar.set_description(f"loss: {loss:.3f}, acy: {acy:.3f}")