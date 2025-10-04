import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import os
#set proxy for script
# os.environ['HTTP_PROXY']  = "http://245hsbd009%40ibab.ac.in:verma1903@proxy.ibab.ac.in:3128/"
# os.environ['HTTPS_PROXY'] = "http://245hsbd009%40ibab.ac.in:verma1903@proxy.ibab.ac.in:3128/"

# transforms
transform = transforms.Compose([
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# load the CIFAR-10 dataset
training_data = datasets.CIFAR10(
    root='/home/ibab/PycharmProjects/DL-Lab/Lab12/data',
    train=True,
    transform=transform,
    download=False
)

testing_data = datasets.CIFAR10(
    root='/home/ibab/PycharmProjects/DL-Lab/Lab12/data',
    train=False,
    transform=transform,
    download=False
)

# dataloaders
train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(testing_data, batch_size=64, shuffle=False)

##use RESnet18 model with pre-trained weights for feature extraction
model = models.resnet18(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))

# freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# initialize the new output layer for 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.4f}")

    # evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch + 1}, Test Accuracy: {(100 * correct / total):.2f}%")