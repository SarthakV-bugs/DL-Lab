import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(0)


# Define the generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return torch.tanh(self.fc3(x))


# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        return self.sigmoid(self.fc3(x))


# Hyperparameters
latent_dim = 10
hidden_dim = 128
output_dim = 1
batch_size = 64
num_epochs = 5000
learning_rate = 0.0002

# Create generator and discriminator instances
generator = Generator(latent_dim, hidden_dim, output_dim)
discriminator = Discriminator(output_dim, hidden_dim)

# Define loss functions and optimizers
criterion = nn.BCELoss()  # Binary cross-entropy loss
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)


# Create a function to generate random noise vectors
def sample_noise(batch_size, latent_dim):
    return torch.randn(batch_size, latent_dim)


# Training loop
for epoch in range(num_epochs):
    # Generate random noise and real data batches
    z = sample_noise(batch_size, latent_dim).to(torch.float32)  # Noise input to generator
    real_data = torch.randn(batch_size, output_dim).to(torch.float32)  # Real Gaussian data as target

    # Train the discriminator with real and fake data
    d_optimizer.zero_grad()
    real_output = discriminator(real_data)
    real_loss = criterion(real_output, torch.ones_like(real_output))  # Discriminator wants to classify real as true (1)

    fake_data = generator(z).detach()  # Generate fake data from noise input
    fake_output = discriminator(fake_data)
    fake_loss = criterion(fake_output,
                          torch.zeros_like(fake_output))  # Discriminator wants to classify fake as false (0)

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()

    # Train the generator
    g_optimizer.zero_grad()
    z = sample_noise(batch_size, latent_dim).to(torch.float32)  # Noise input to generator
    fake_data = generator(z)
    gen_output = discriminator(fake_data)
    g_loss = criterion(gen_output,
                       torch.ones_like(gen_output))  # Generator wants discriminator to classify its output as true (1)

    g_loss.backward()
    g_optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Visualization of real and generated data
with torch.no_grad():
    test_noise = sample_noise(100, latent_dim).to(torch.float32)
    generated_data = generator(test_noise).numpy()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Real Data")
plt.hist(real_data.numpy(), bins=30, density=True, alpha=0.5)
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.title("Generated Data")
plt.hist(generated_data.flatten(), bins=30, density=True, alpha=0.5)
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.show()