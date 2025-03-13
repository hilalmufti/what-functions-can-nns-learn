import functools
import random

from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

"""From a Workshop: https://colab.research.google.com/drive/1YFKhgbDXjBXHf9ev7jN57XToiibCNqiz?usp=sharing#scrollTo=f5rq6JMmQEOi"""

@functools.lru_cache(maxsize=None)
def primes_upto(N: int) -> List[int]:
    """
    Return the prime numbers in the range [0, N) in increasing order.
    """

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


def omega_upto(N: int) -> Tuple[List[int], List[int]]:
    """
    The little-omega of n is the number of distinct prime factors of n, and the big-omega of n is the
    number of prime factors of n counting multiplicity: https://en.wikipedia.org/wiki/Prime_omega_function.

    The function omega_upto returns two parallel arrays of N elements, giving the little and big omegas of [0, N).
    """

    little = [0] * N
    big = [0] * N

    for p in primes_upto(N):
        # For each prime, mark off its multiples in the little array.
        for i in range(p, N, p):
            little[i] += 1

        # For each prime power, mark off its multiples in the big array.
        power = p
        while power < N:
            for i in range(power, N, power):
                big[i] += 1
            power *= p

    return little, big


def mobius_upto(N: int) -> List[int]:
    """
    The mobius function of n is::

        +1 if n has an even number of prime factors, all distinct,
        -1 if n has an odd number of prime factors, all distinct, and
         0 if n has a repeated prime factor.

    We define the mobius function at 0 to be equal to 1.
    """

    little_omega, big_omega = omega_upto(N)
    return [
        0 if little_omega[i] != big_omega[i] else 1 if little_omega[i] % 2 == 0 else -1
        for i in range(N)
    ]

def binary_block(bits: int):
    """
    Return a tensor T of shape (2^bits, bits) such that T[i, :] is a 01-vector
    of length bits, containing the binary representation of i. For example:
    """
    # Matrix of coordinates 100, 010, 001 etc.
    coords = np.identity(bits, dtype=int)

    # Add 0 or 1 lots of coords in every way possible.
    result = np.zeros(bits, dtype=int)
    for i in reversed(range(bits)):
        result = np.vstack([result, result + coords[:,i]])

    return result


def create_dataset(bits: int) -> Tuple[int, int, np.array, np.array]:
    mu = mobius_upto(2 ** bits)
    mu_input = torch.tensor(binary_block(bits), dtype=torch.float32)
    mu_output = torch.tensor(mu) + 1

    return mu, mu_input, mu_output


def create_network(layers: List[int]):
    return nn.Sequential(*[
        module
        for x, y in zip(layers, layers[1:])
        for module in [nn.Linear(x, y), nn.ReLU()]
    ][:-1])

# Set the bitsize and create the dataset.
BITS = 16
mu, mu_input, mu_output = create_dataset(BITS)
batch_size = 128

# Select 50% of the data to be training data.
indices = torch.randperm(2**BITS)
training_idx, validation_idx = indices[:2**BITS // 2], indices[2**BITS // 2:]
mu_input_training, mu_output_training = mu_input[training_idx], mu_output[training_idx]
mu_input_validation, mu_output_validation = mu_input[validation_idx], mu_output[validation_idx]

def create_and_train_binary_model(hidden_layers: List[int], epochs: int):
    model = create_network([BITS] + hidden_layers + [3])
    loss_function = nn.CrossEntropyLoss()
    # optimiser = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.01)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # We will record the loss functions across the dataset every 100 epochs.
    LOSS_INTERVAL = 100
    recorded_loss = torch.full(size=(epochs//LOSS_INTERVAL, 3), fill_value=0.0)

    # At each epoch, show the network 10 data points and do gradient descent.
    for epoch in range(epochs):
        training_size = mu_input_training.shape[0]
        batch = torch.tensor(random.sample(range(training_size), k=batch_size))

        optimiser.zero_grad()
        model_output = model(mu_input_training[batch])
        loss = loss_function(model_output, mu_output_training[batch])
        loss.backward()
        optimiser.step()

        if epoch % LOSS_INTERVAL == 0:
            # Record the training and validation loss of the network at each epoch.
            # We do this with no_grad enabled: this disables computation tracking and uses much less RAM.
            with torch.no_grad():
                recorded_loss[epoch // LOSS_INTERVAL, 0] = epoch
                recorded_loss[epoch // LOSS_INTERVAL, 1] = loss_function(model(mu_input_training), mu_output_training)
                recorded_loss[epoch // LOSS_INTERVAL, 2] = loss_function(model(mu_input_validation), mu_output_validation)

    return model, recorded_loss

hidden_layers = [10, 10]
binary_model, loss = create_and_train_binary_model(hidden_layers, epochs=1000)

# Plot the loss function over time.
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(loss[:, 0], loss[:, 1], label='Training loss')
ax.plot(loss[:, 0], loss[:, 2], label='Validation loss')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid()
ax.set_title(f"Loss over time")
plt.show()

with torch.no_grad():
  results = pd.DataFrame.from_dict({
      'n': range(2**BITS),
      'actual': mu,
      'predicted': (torch.max(binary_model(mu_input), dim=1).indices - 1),
  })

# print(results)

with torch.no_grad():
    predictions = torch.max(binary_model(mu_input), dim=1).indices - 1  # Convert predictions to Möbius values
    correct = (predictions == torch.tensor(mu)).sum().item()
    total = len(mu)
    accuracy = correct / total

print(f"Model Accuracy: {accuracy:.4f}")

with torch.no_grad():
    predictions = torch.max(binary_model(mu_input), dim=1).indices - 1  # Convert predictions to Möbius values
    actual_values = torch.tensor(mu)

    # Accuracy for predicting -1
    mask_neg1 = (actual_values == -1)
    correct_neg1 = (predictions[mask_neg1] == actual_values[mask_neg1]).sum().item()
    total_neg1 = mask_neg1.sum().item()
    accuracy_neg1 = correct_neg1 / total_neg1 if total_neg1 > 0 else 0

    # Accuracy for predicting 1
    mask_pos1 = (actual_values == 1)
    correct_pos1 = (predictions[mask_pos1] == actual_values[mask_pos1]).sum().item()
    total_pos1 = mask_pos1.sum().item()
    accuracy_pos1 = correct_pos1 / total_pos1 if total_pos1 > 0 else 0

print(f"Accuracy for predicting -1: {accuracy_neg1:.4f}")
print(f"Accuracy for predicting 1: {accuracy_pos1:.4f}")

with torch.no_grad():
    predictions = torch.max(binary_model(mu_input), dim=1).indices - 1  # Convert predictions to Möbius values
    actual_values = torch.tensor(mu)

    # Accuracy for predicting 0
    mask_0 = (actual_values == 0)
    correct_0 = (predictions[mask_0] == actual_values[mask_0]).sum().item()
    total_0 = mask_0.sum().item()
    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0

    # Accuracy for predicting ±1
    mask_1 = (actual_values != 0)  # Covers both -1 and 1
    correct_1 = (predictions[mask_1] == actual_values[mask_1]).sum().item()
    total_1 = mask_1.sum().item()
    accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0

print(f"Accuracy for predicting 0: {accuracy_0:.4f}")
print(f"Accuracy for predicting ±1: {accuracy_1:.4f}")
