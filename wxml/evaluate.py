import torch

from wxml.numbers import bitround

def accuracy(x, y):
    acc = torch.sum(x == y) / len(x)
    return acc.item()

def evaluate(model, loss_fn, loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_acc = 0
    count = 0
    
    with torch.no_grad():  # Disable gradient computation
        for x, y in loader:
            # print("test:", model(x))
            # y_pred = [round(tt) for tt in model(x)] # round to the nearest integer 
            #print(round(model(x)[0][0]))
            y_pred = torch.tensor([bitround(tt) for tt in model(x)])
            print(y_pred)
            loss = loss_fn(y_pred, y)
            acc = accuracy(y_pred, y)
            
            total_loss += loss.item() * len(x)
            total_acc += acc * len(x)
            count += len(x)
    
    avg_loss = total_loss / count
    avg_acc = total_acc / count
    
    print(f"Test Loss: {avg_loss:.3f}, Test Accuracy: {avg_acc:.3f}")
    return avg_loss, avg_acc