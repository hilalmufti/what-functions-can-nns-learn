import pandas as pd
import matplotlib.pyplot as plt

# Load experiment results from the file
results_file = "MobiusFunction_experiment_results.txt"
function_name = "MobiusFunction_withMoreLayers"

def load_results(file_path, start_line=1, end_line=27):
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()[start_line-1:end_line]  # Select the range of lines

    for line in lines:
        parts = line.strip().split(" | ")
        N = int(parts[0].split("=")[1].split(",")[0])
        input_type = parts[0].split(", ")[1].split("=")[1]
        layers = int(parts[0].split(", ")[2].split("=")[1])
        avg_loss = float(parts[1].split(": ")[1])
        accuracy = float(parts[2].split(": ")[1])
        data.append([N, input_type, layers, avg_loss, accuracy])

    return pd.DataFrame(data, columns=["N", "Input Type", "Layers", "Avg Loss", "Accuracy"])


def plot_accuracy_vs_hidden_layers(df, function_name):
    plt.figure(figsize=(10, 6))
    for input_type in df["Input Type"].unique():
        subset = df[df["Input Type"] == input_type]
        plt.scatter(subset["Layers"], subset["Accuracy"], marker='o', label=f"Input={input_type}")

    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Hidden Layers for Different Input Types")
    plt.legend()
    plt.grid(True)
    plt.show


def plot_loss_vs_hidden_layers(df, function_name):
    plt.figure(figsize=(10, 6))
    for input_type in df["Input Type"].unique():
        subset = df[df["Input Type"] == input_type]
        plt.scatter(subset["Layers"], subset["Avg Loss"], marker='o', label=f"Input={input_type}")

    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Average Loss")
    plt.title("Loss vs. Hidden Layers for Different Input Types")
    plt.legend()
    plt.grid(True)
    plt.show

def plot_accuracy_vs_input_type(df, function_name):
    plt.figure(figsize=(10, 6))
    for N in df["N"].unique():
        subset = df[df["N"] == N]
        plt.scatter(subset["Input Type"], subset["Accuracy"], marker='o', label=f"N={N}")

    plt.xlabel("Input Type")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Input Type for Different Dataset Sizes")
    plt.legend()
    plt.grid(True)
    plt.show


# Load the data
df = load_results(results_file, start_line=28, end_line=72)

# Generate plots
plot_accuracy_vs_input_type(df, function_name)
# plot_accuracy_vs_hidden_layers(df, function_name)
# plot_loss_vs_hidden_layers(df, function_name)