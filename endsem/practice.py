# #denoising autoencoder from scratch
# import torch
# from matplotlib.pyplot import plot
# from torch import nn
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader
# from torchvision import transforms, datasets
#
# #define the denoising architecture
# class Denoising_autoencoder(nn.Module):
#     def __init__(self,input_size,hidden_size,bottleneck_dim, output_size):
#         super(Denoising_autoencoder, self).__init__()
#
#         #encoder block
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size), #input values are b/w 0 and 1
#             nn.ReLU(),
#             nn.Linear(hidden_size, bottleneck_dim),
#             nn.ReLU()
#             )
#
#         #decoder block
#         self.decoder = nn.Sequential(
#             nn.Linear(bottleneck_dim, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size),
#             nn.Sigmoid() #values between 0 and 1
#         )
#
#     def forward(self, x):
#         enocoded = self.encoder(x)
#         x = self.decoder(enocoded)
#         return x
#
# ###IMP: for a denoising autoencoder we need to add some form of noise into the input data
# def add_noise(inputs, noise_factor=0.5):
#     noise = torch.randn_like(inputs) * noise_factor
#     noisy_inputs = inputs + noise
#     return torch.clamp(noisy_inputs, 0., 1.)
#
# #train and evaluation loop
#
# def train_autoencoder(model, train_loader, epochs):
#     total_loss = 0
#     model.train()
#     for epoch in range(epochs):
#         for batch_idx, (data, target) in enumerate(train_loader):
#
#             #create noisy version of the data
#             noisy_data = add_noise(data, noise_factor=0.5)
#
#             #define the criterion, use mse
#             criterion = nn.MSELoss()
#
#             optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#             optimizer.zero_grad()
#
#             #output
#             output = model(noisy_data)
#             loss = criterion(output, noisy_data)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#         print(f'epoch {epoch + 1}, loss: {total_loss / len(train_loader)}')
#
#
#
# def evaluate_autoencoder(model, test_loader):
#     total_loss = 0
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#
#             noisy_data = add_noise(data, noise_factor=0.5)
#             criterion = nn.MSELoss()
#             reconstructed = model(noisy_data)
#             loss = criterion(reconstructed, noisy_data)
#             total_loss += loss.item()
#
#
# def main():
#     # load the dataset
#     transform = transforms.Compose([transforms.ToTensor()])
#
#     train_dataset = datasets.MNIST(root='/home/ibab/Desktop/DL_datasets/', train=True, transform=transform)
#     test_dataset = datasets.MNIST(root='/home/ibab/Desktop/DL_datasets/', train=False, transform=transform)
#
#     img, label = next(iter(test_dataset))
#     print(img.shape)
#
#     # dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
# if __name__ == '__main__':
#     main()


#
# Train a GAN model on a simulated dataset of random noise samples that follow a 1D Gaussian distribution. The goal is for the generator to learn how to produce samples that resemble real Gaussian-distributed data, while the discriminator learns to distinguish between real and generated samples. Use the following architecture and hyperparameters: the generator should consist of Linear–ReLU–Linear–ReLU–Linear layers, and the discriminator should consist of Linear–LeakyReLU–Linear–LeakyReLU–Linear–Sigmoid layers. Use a latent vector of dimension 10, a hidden dimension of 128, an output dimension of 1, a batch size of 64, 5000 training epochs, and a learning rate of 0.0002. After training, plot the real and generated data to visually compare their distributions.
#
# I used the following code to train the GANs:

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Generator
gan = GAN()
gan.train()
gan.cuda()

# Discriminator
disc = Discriminator()
disc.train()
disc.cuda()

# Optimizers
gan_optimizer = optim.Adam(gan.parameters(), lr=0.0002)
disc_optimizer = optim.Adam(disc.parameters(), lr=0.0002)

# Training data
data = torch.randn(1000, 1).cuda()

# Loss functions
gan_loss = nn.BCELoss()
disc_loss = nn.BCELoss()

# Training
for epoch in range(5000):
    gan.zero_grad()
    disc.zero_grad()

    # Generator
    z = Variable(torch.randn(64, 10).cuda())
    fake = gan(z)
    loss = gan_loss(fake, data)
    loss.backward()
    gan_optimizer.step()

    # Discriminator
    real = Variable(torch.zeros(64, 1).cuda())
    fake = gan(z)
    loss = disc_loss(fake, real)
    loss.backward()
    disc_optimizer.step()

    if epoch % 100 == 0:
        print('epoch: %d, loss: %.3f' % (epoch, loss.data[0]))

# Generated data
fake = gan(Variable(torch.randn(1000, 1).cuda()))

# Real data
real = Variable(data.cuda())

# Plot
plt.plot(np.arange(1000), real.data.cpu().numpy().reshape(1000), 'bo')
plt.plot(np.arange(1000), fake.data.cpu().numpy().reshape(1000), 'ro')
plt.legend(['Real', 'Generated'])
plt.show()
