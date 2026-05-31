import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Data
z = np.linspace(-10, 10, 100)

# --- Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# --- Plotting ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot Sigmoid
axs[0].plot(z, sigmoid(z), label='Sigmoid', color='blue')
axs[0].plot(z, sigmoid_derivative(z), label='Derivative', color='red', linestyle='--')
axs[0].set_title("Sigmoid")
axs[0].legend()
axs[0].grid(True)

# Plot Tanh
axs[1].plot(z, tanh(z), label='Tanh', color='blue')
axs[1].plot(z, tanh_derivative(z), label='Derivative', color='red', linestyle='--')
axs[1].set_title("Tanh")
axs[1].legend()
axs[1].grid(True)

# Plot ReLU
axs[2].plot(z, relu(z), label='ReLU', color='blue')
axs[2].plot(z, relu_derivative(z), label='Derivative', color='red', linestyle='--')
axs[2].set_title("ReLU")
axs[2].legend()
axs[2].grid(True)

plt.show()

#A2

import torch
import torchvision
import torchvision.transforms as transforms

# Define transform: ToTensor converts 0-255 pixels to [0, 1] floats
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load Dataset (Train set)
dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Verify one sample
image, label = dataset[0]
print(f"Min pixel value: {image.min()}")
print(f"Max pixel value: {image.max()}")
print(f"Image shape: {image.shape}")

#A3)

import torch
import torch.nn as nn

# Define the single layer (3 inputs -> 1 output)
model = nn.Linear(in_features=3, out_features=1)

# Generate a random input sample (Batch size 1, 3 features)
input_data = torch.randn(1, 3)

# Perform forward pass
output = model(input_data)

print("Input:", input_data)
print("Output:", output)

#A4)
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder: 784 -> 128 -> 32
        # Instruction: Linear-ReLU-Linear-ReLU
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        # Decoder: 32 -> 128 -> 784
        # Instruction: Linear-ReLU-Linear-ReLU
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# 1. Instantiate Model
model = Autoencoder()

# 2. Generate random input X (784-dimensional vector)
X = torch.randn(1, 784)

# 3. Generate Latent Representation
# We only need the forward pass, no training required per prompt
_, latent_representation = model(X)

print(f"Input shape: {X.shape}")
print(f"Latent representation shape: {latent_representation.shape}")  # Should be [1, 32]
print("\nLatent Representation Values:\n", latent_representation)