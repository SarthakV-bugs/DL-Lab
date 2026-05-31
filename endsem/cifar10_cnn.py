import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Load and Normalize CIFAR10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# 2. Define the Model: CONV-CONV-MAXPOOL-LINEAR-LINEAR
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Conv Layer 1: 3 input channels -> 32 output channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Conv Layer 2: 32 input -> 64 output
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # MaxPool: Reduces 32x32 to 16x16
        self.pool = nn.MaxPool2d(2, 2)

        # Flatten size: 64 channels * 16 * 16 pixels = 16384
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 64 * 16 * 16)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

# 3. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 4. Train (5 Epochs)
print("Starting Training...")
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')

# 5. Evaluate
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')