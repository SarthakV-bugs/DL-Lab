"""Denoising autoencoder"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='/home/ibab/Desktop/DL_datasets/', train=True,  transform=transform)
test_dataset = datasets.MNIST(root='/home/ibab/Desktop/DL_datasets/', train=False,  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=784, bottleneck_dim=128):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim),
            nn.ReLU()  # Compressed representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Outputs values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def add_noise(inputs, noise_factor=0.5):
    """Adds Gaussian noise to the inputs."""
    noise = torch.randn_like(inputs) * noise_factor
    noisy_inputs = inputs + noise
    # Clip to keep values valid (0,1)
    return torch.clamp(noisy_inputs, 0., 1.)


def train_autoencoder(model, train_loader, epochs=5):
    criterion = nn.MSELoss()
    # Adam usually converges faster than SGD for Autoencoders
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            img = img.view(-1, 784)

            # FIX 2: Create noisy version for input
            noisy_img = add_noise(img)

            optimizer.zero_grad()

            # Forward pass: Input is NOISY
            outputs = model(noisy_img)

            # Loss calculation: Target is CLEAN (img), not noisy_img
            loss = criterion(outputs, img)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')


def eval_autoencoder(model, test_loader):
    model.eval()

    # Get one batch for visualization
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images.view(-1, 784)

    # Add noise to test images to see if model removes it
    noisy_images = add_noise(images)

    with torch.no_grad():
        reconstructed = model(noisy_images)

    # Visualization
    num_images = 6
    fig, axes = plt.subplots(3, num_images, figsize=(15, 6))

    # Rows: Original -> Noisy Input -> Reconstructed
    row_titles = ["Original", "Noisy Input", "Reconstructed"]

    for i in range(num_images):
        # Original
        axes[0, i].imshow(images[i].view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title(row_titles[0])

        # Noisy
        axes[1, i].imshow(noisy_images[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title(row_titles[1])

        # Reconstructed
        axes[2, i].imshow(reconstructed[i].view(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title(row_titles[2])

    plt.tight_layout()
    plt.show()


# Run the experiment
bottleneck_dim = 64
print(f"Training with bottleneck: {bottleneck_dim}")
model = DenoisingAutoencoder(bottleneck_dim=bottleneck_dim)
train_autoencoder(model, train_loader)
eval_autoencoder(model, test_loader)