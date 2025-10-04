import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 1. Simulate Vanishing Gradient Problem
class VanishingGradientNet(nn.Module):
    """Deep network with sigmoid activation - causes vanishing gradients"""

    def __init__(self, num_layers=10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.Sigmoid())  # Sigmoid causes vanishing gradients
        layers.append(nn.Linear(100, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 2. Simulate Exploding Gradient Problem
class ExplodingGradientNet(nn.Module):
    """Deep network with poor weight initialization - causes exploding gradients"""

    def __init__(self, num_layers=10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layer = nn.Linear(100, 100)
            # Bad initialization: large weights cause exploding gradients
            nn.init.normal_(layer.weight, mean=0, std=2.0)
            layers.append(layer)
            layers.append(nn.Tanh())
        layers.append(nn.Linear(100, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Function to track gradients during training
def track_gradients(model, data, target, criterion):
    """Calculate gradients and return gradient norms for each layer"""
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    gradient_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad_norm = param.grad.norm().item()
            gradient_norms.append(grad_norm)

    return gradient_norms


# Function to plot gradient values across layers
def plot_gradients(gradient_history, title):
    """Plot gradient norms across layers over training iterations"""
    plt.figure(figsize=(10, 6))
    gradient_history = np.array(gradient_history)

    for layer_idx in range(gradient_history.shape[1]):
        plt.plot(gradient_history[:, layer_idx], label=f'Layer {layer_idx + 1}')

    plt.xlabel('Training Iteration')
    plt.ylabel('Gradient Norm')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Training function
def train_and_visualize(model, model_name, num_iterations=50):
    """Train model and visualize gradient problem"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    gradient_history = []

    print(f"\nTraining {model_name}...")
    for i in range(num_iterations):
        # Generate random data
        data = torch.randn(32, 100)
        target = torch.randint(0, 10, (32,))

        # Track gradients before optimizer step
        grad_norms = track_gradients(model, data, target, criterion)
        gradient_history.append(grad_norms)

        # Training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{num_iterations}, Loss: {loss.item():.4f}")
            print(f"  First layer gradient norm: {grad_norms[0]:.6f}")
            print(f"  Last layer gradient norm: {grad_norms[-1]:.6f}")

    return gradient_history


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Lab 9: Vanishing & Exploding Gradient Problem")
    print("=" * 60)

    # 1. Vanishing Gradient Problem
    print("\n1. VANISHING GRADIENT PROBLEM")
    print("-" * 60)
    vanishing_model = VanishingGradientNet(num_layers=10)
    vanishing_grads = train_and_visualize(vanishing_model, "Vanishing Gradient Network")
    plot_gradients(vanishing_grads, "Vanishing Gradient Problem (Sigmoid Activation)")

    # 2. Exploding Gradient Problem
    print("\n2. EXPLODING GRADIENT PROBLEM")
    print("-" * 60)
    exploding_model = ExplodingGradientNet(num_layers=10)
    exploding_grads = train_and_visualize(exploding_model, "Exploding Gradient Network")
    plot_gradients(exploding_grads, "Exploding Gradient Problem (Poor Initialization)")