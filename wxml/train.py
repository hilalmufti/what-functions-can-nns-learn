import torch
# import tqdm
from tqdm.notebook import tqdm, trange


def accuracy(x, y):
    acc = torch.sum(x == y) / len(x)
    return acc.item()


def count_matches(x, y):
    return (~(x.bool() ^ y.bool())).float().sum().item()


def classification_accuracy(loader, model, device, dtype):
    # if loader.dataset.train:
    #     print("Checking accuracy on validation set")
    # else:
    #     print("Checking accuracy on test set")
    n_correct = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            scores = model(x)
            preds = scores.round()
            # preds = torch.argmax(scores, dim=-1, keepdim=True)
            # n_correct += torch.sum(preds == y).item()
            n_correct += count_matches(preds, y)
            # print(count_matches(preds, y), preds, y)
            # n += len(y)
            n += len(y) * y.shape[-1]
        acy = float(n_correct) / n # TODO: I don't think you need to convert to float
        # print(f"{n_correct} / {n} correct ({acy:.2f})")
    return acy


def val_step(loader, model, loss_fn, device, dtype):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            scores = model(x)
            loss += loss_fn(scores, y).item()
    return loss / len(loader)



def multiclassifcation_accuracy(loader, model, device, dtype):
    ...



def train_step(model, opt, loss_fn, x, y, device, dtype):
    assert x.requires_grad and y.requires_grad
    model.train()

    n_correct = 0
    n = 0

    x = x.to(device, dtype=dtype)
    y = y.to(device, dtype=dtype)
    scores = model(x)
    loss = loss_fn(scores, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    preds = scores.round()
    n_correct += count_matches(preds, y)
    n += len(y) * y.shape[-1]
    acy = float(n_correct) / n

    return scores, loss.item(), acy


def train(model, loss_fn, opt, train_loader, val_loader, device, dtype, epochs=10, print_every=100):
    results = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": []
    }
    model = model.to(device)
    # for e in (pbar := tqdm.trange(epochs, desc="loss: inf, acc: inf")):
    # with tqdm.trange(epochs, desc="Training") as pbar:
    with trange(epochs) as pbar:
        for e in pbar:
            epoch_results = {
                "train_losses": [],
                "train_accuracies": [],
            }
            for i, (x, y) in enumerate(train_loader):
                scores, loss, acy = train_step(model, opt, loss_fn, x, y, device, dtype)

                epoch_results["train_losses"].append(loss)
                epoch_results["train_accuracies"].append(acy)

                if i % print_every == 0:
                    pbar.set_postfix({"loss": f"{loss:.4f}", "accuracy": f"{acy:.4f}"})

            
            train_loss = sum(epoch_results["train_losses"]) / len(epoch_results["train_losses"])
            train_acy = sum(epoch_results["train_accuracies"]) / len(epoch_results["train_accuracies"])
            val_loss = val_step(val_loader, model, loss_fn, device, dtype)
            val_acy = classification_accuracy(val_loader, model, device, dtype)
            results["train_losses"].append(train_loss)
            results["val_losses"].append(val_loss)
            results["train_accuracies"].append(train_acy)
            results["val_accuracies"].append(val_acy)

            tqdm.write(f"epoch={e} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | train_acy={train_acy:.4f} | val_acy={val_acy:.4f}")



            # if i % print_every == 0:
            #     train_acy = classification_accuracy(train_loader, model, device, dtype)
            #     val_acy = classification_accuracy(val_loader, model, device, dtype)
            #     val_loss = val_step(val_loader, model, loss_fn, device, dtype)
            #     results["train_losses"].append(loss)
            #     results["val_losses"].append(val_loss)
            #     results["train_accuracies"].append(train_acy)
            #     results["val_accuracies"].append(val_acy)
            #     print(f"i={i} | train_loss={loss:.4f} | val_loss={val_loss:.4f} | train_acy={train_acy:.4f} | val_acy={val_acy:.4f}")
        
    return results