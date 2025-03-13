import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sympy import mobius, factorint
import matplotlib.pyplot as plt
import os
import yaml

# Import the MLP and training functions
from wxml.mlp import MLP
from wxml.train import train

# Load configurations from config.yaml
def load_config(config_path=None):
    if config_path is None:
        # Adjust the path to where your config.yaml is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "../config.yaml")  # Modify path as needed

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["mobius"]["args"]


# get loss function
def get_loss_function(loss_name):
    loss_functions = {
        "binary_crossentropy": nn.BCELoss(),
        "cross_entropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "huber": nn.SmoothL1Loss(),
    }

    if loss_name not in loss_functions:
        raise ValueError(f"Unsupported loss function: {loss_name}. Choose from {list(loss_functions.keys())}")

    return loss_functions[loss_name]

# get optimizer
def get_optimizer(optimizer_name, model, lr):
    optimizers = {
        "adam": optim.Adam(model.parameters(), lr=lr),
        "sgd": optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "adamw": optim.AdamW(model.parameters(), lr=lr),
        "rmsprop": optim.RMSprop(model.parameters(), lr=lr),
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from {list(optimizers.keys())}")

    return optimizers[optimizer_name]


# Compute Möbius function values
def compute_mobius(n):
    return int(mobius(n))

# Convert integer to binary representation
def to_binary(n, bits=14):
    return np.array([int(x) for x in bin(n)[2:].zfill(bits)])

# get primes up to N by Sieve of Eratosthenes
# primes_upto(12) = [2, 3, 5, 7, 11]
def primes_upto(N):
    # N should be at least 2 to run the sieve below.
    if N <= 1:
        return []
    # Sieve of Eratosthenes, using an array of integers of 0 (not prime) or 1 (prime).
    sieve = [1] * N
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, N):
        if sieve[i] == 1:
            for j in range(i*i, N, i):
                sieve[j] = 0

    return [p for p in range(N) if sieve[p] == 1]

# Compute modular features
#  modular_features(10, 12) --> [0, 1, 0, 3, 10]
def modular_features(n, N):
    return np.array([n % p for p in primes_upto(N)])

# MLP Model for Möbius classification
class MobiusMLP(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(MobiusMLP, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, 3))  # Output classes (-1, 0, 1)
        layers.append(nn.Softmax(dim=1))  # Multi-class classification
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def run_experiment(config, input_type, hidden_layer):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    N = config["n_samples"]
    modular_features_N = config["modular_features_N"]
    output_file = config["output_file"]

    X = np.arange(1, N).reshape(-1, 1)
    y = np.array([compute_mobius(n) + 1 for n in X.flatten()], dtype=np.int64)  # Shift labels to (0,1,2)

    if input_type == "binary":
        X = np.array([to_binary(n) for n in X.flatten()])
    elif input_type == "modular_features":
        X = np.array([modular_features(n, modular_features_N) for n in X.flatten()])
    else:
        X = X / N  # Normalize integer inputs

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config["seed"])

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Use long for classification
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model, loss, optimizer
    model = MobiusMLP(X_train.shape[1], hidden_layer)
    loss_fn = get_loss_function(config["loss"])
    optimizer = get_optimizer(config["optimizer"], model, float(config["lr"]))

    # Tracking loss and accuracy
    train_losses, test_losses, test_accuracies = [], [], []

    epochs = config["epochs"]
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation
        model.eval()
        correct, total, total_test_loss = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                total_test_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = correct / total
        test_accuracies.append(accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {accuracy:.4f}")

    # Save results to file
    result = f"N={N}, Input={input_type}, Layers={len(hidden_layer)} | Avg Train Loss: {total_loss / (epochs * len(train_loader)):.4f} | Test Accuracy: {accuracy:.4f}\n"
    with open(output_file, "a") as f:
        f.write(result)

    # Plot results
    plot_loss_accuracy(train_losses, test_losses, test_accuracies, input_type, hidden_layer, N)


def plot_loss_accuracy(train_losses, test_losses, test_accuracies, input_type, hidden_layers, N):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linestyle="-", marker="o")
    # plt.plot(test_losses, label="Test Loss", linestyle="-", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve: {input_type}, {len(hidden_layers)} Layers, {N} Samples")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Test Accuracy: {input_type}, {len(hidden_layers)} Layers, {N} Samples")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Run experiments
input_types = ['int', 'binary', 'modular']
hidden_layers = [[128, 64, 32]]
config = load_config()

for input_type in input_types:
    for hidden_layer in hidden_layers:
        run_experiment(config, input_type, hidden_layer)