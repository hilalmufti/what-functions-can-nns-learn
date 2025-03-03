import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sympy import mobius, factorint

# Compute Möbius function values
def compute_mobius(n):
    return int(mobius(n))  # Ensure integer output

# Convert integer to binary representation
def to_binary(n, bits=14):
    return np.array([int(x) for x in bin(n)[2:].zfill(bits)])

# Compute x mod p for small primes (2, 3, 5, 7, 11)
def modular_features(n):
    return np.array([n % p for p in [2, 3, 5, 7, 11]])

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

# Experiment loop
def run_experiment(N, input_type, hidden_layers, output_file):
    # Generate dataset
    X = np.arange(1, N).reshape(-1, 1)
    y = np.array([compute_mobius(n) + 1 for n in X.flatten()], dtype=np.int64)  # Shift labels to (0,1,2)

    if input_type == 'binary':
        X = np.array([to_binary(n) for n in X.flatten()])
    elif input_type == 'modular':
        X = np.array([modular_features(n) for n in X.flatten()])
    else:  # Default: integer inputs
        X = X / N  # Normalize

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Use long for classification
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer
    model = MobiusMLP(X_train.shape[1], hidden_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training Loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        accuracy = correct / total

    result = f"N={N}, Input={input_type}, Layers={len(hidden_layers)} | Avg Loss: {total_loss / (epochs * len(train_loader)):.4f} | Accuracy: {accuracy:.4f}\n"
    with open(output_file, "a") as f:
        f.write(result)

# Run experiments
Ns = [100, 1000, 10000]
input_types = ['int', 'binary', 'modular']
hidden_layer_configs = [[], [128], [128, 64]]
output_file = "MobiusFunction_experiment_results.txt"

# for N in Ns:
#     for input_type in input_types:
#         for hidden_layers in hidden_layer_configs:
#             run_experiment(N, input_type, hidden_layers, output_file)