import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

def print_sample_results(inputs, predictions, targets, n_samples=5):
    for i in range(min(n_samples, len(inputs))):
        print(f"Input: {inputs[i]}")
        print(f"Predicted: {predictions[i]}")
        print(f"Actual: {targets[i]}")
        print("-" * 50)
