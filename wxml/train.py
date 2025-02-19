import torch
import tqdm


def accuracy(x, y):
    acc = torch.sum(x == y) / len(x)
    return acc.item()


# @torch.compile
def train_step(model, loss_fn, opt, x, y):
    assert x.requires_grad and y.requires_grad
    model.train()
    opt.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    opt.step()
    return y_pred, loss.item()


def train(model, loss_fn, opt, train_loader, val_loader, epochs=10, acy_fn=accuracy):
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in (pbar := tqdm.trange(epochs, desc="loss: inf, acc: inf")):
        for i, (x, y) in enumerate(train_loader):
            y_pred, loss = train_step(model, loss_fn, opt, x, y)
            losses.append(loss)
            if i % 100 == 0:
                acy = accuracy(y_pred, y)
                accuracies.append(acy)
                pbar.set_description(f"loss: {loss:.3f}, acy: {acy:.3f}")
        
        with torch.no_grad():
            model.eval()
            for x, y in val_loader:
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_losses.append(loss.item())
                val_accuracies.append(accuracy(y_pred, y))
        
        

    return losses, accuracies, val_losses, val_accuracies