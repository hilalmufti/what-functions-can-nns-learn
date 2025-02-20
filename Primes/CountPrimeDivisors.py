import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# count number of distinct prime divisors
def count_prime_divisors(n):
    if isinstance(n, np.ndarray):  # Ensure n is a scalar integer
        n = int(n.item())
    prime_factors = set()
    while n % 2 == 0:
        prime_factors.add(2)
        n //= 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            prime_factors.add(i)
            n //= i
    if n > 2:
        prime_factors.add(n)
    return len(prime_factors)


# convert a number to binary representation
def to_binary(n, bits=14):  # 14 bits to cover numbers up to 10,000
    return np.array([int(x) for x in bin(n)[2:].zfill(bits)])


# compute modular features
def modular_features(n):
    return np.array([n % p for p in [2, 3, 5, 7, 11]])


# Define MLP Model
class PrimeMLP(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(PrimeMLP, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        # layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Experiment loop
def run_experiment(N, input_type, hidden_layers, output_file):
    # Generate dataset
    X = np.arange(2, N).reshape(-1, 1)
    y = np.array([count_prime_divisors(n) for n in X.flatten()]).reshape(-1, 1)

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
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer
    model = PrimeMLP(X_train.shape[1], hidden_layers)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training Loop
    epochs = 40
    total_loss = 0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    avg_train_loss = total_loss / (epochs * len(train_loader))

    # Evaluation
    model.eval()
    total_eval_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_eval_loss += loss.item()
            predicted = torch.round(outputs)  # Round to nearest integer
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    accuracy = correct / total

    result = f"N={N}, Input={input_type}, Layers={len(hidden_layers)} | Avg Train Loss: {avg_train_loss:.4f} | Accuracy: {accuracy:.4f}\n"
    with open(output_file, "a") as f:
        f.write(result)


# Run experiments
output_file = "CountPrimeDivisors_experiment_results.txt"
Ns = [100, 1000, 10000]
input_types = ['int', 'binary', 'modular']
hidden_layer_configs = [[], [128], [128, 64]]

for N in Ns:
    for input_type in input_types:
        for hidden_layers in hidden_layer_configs:
            run_experiment(N, input_type, hidden_layers, output_file)
