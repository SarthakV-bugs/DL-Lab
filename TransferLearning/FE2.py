#FE - Feature extraction
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import os

# Set proxy for script
os.environ['HTTP_PROXY'] = "http://245hsbd009%40ibab.ac.in:verma1903@proxy.ibab.ac.in:3128/"
os.environ['HTTPS_PROXY'] = "http://245hsbd009%40ibab.ac.in:verma1903@proxy.ibab.ac.in:3128/"

# Normalize w.r.t. ImageNet dataset
transform_train = transforms.Compose([
    transforms.Resize(224),  # very important for ResNet pretrained
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10
training_data = datasets.CIFAR10(
    root='/home/ibab/PycharmProjects/DL-Lab/Lab12/data',
    train=True,
    transform=transform_train,
    download=False
)

testing_data = datasets.CIFAR10(
    root='/home/ibab/PycharmProjects/DL-Lab/Lab12/data',
    train=False,
    transform=transform_test,
    download=False
)

# Split train into train/val
train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(testing_data, batch_size=128, shuffle=False)

# Load ResNet18 with pretrained weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.fc = nn.Linear(model.fc.in_features, 10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=1e-4)

# Training loop with early stopping
num_epochs = 100
best_val_accuracy = 0.0
patience = 10
patience_counter = 0

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

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    current_val_accuracy = 100 * correct / total

    # Early stopping logic
    if current_val_accuracy > best_val_accuracy:
        best_val_accuracy = current_val_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")  # save best model
    else:
        patience_counter += 1

    # Print every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Acc: {current_val_accuracy:.2f}%, Best: {best_val_accuracy:.2f}%")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Load best model before testing
model.load_state_dict(torch.load("best_model.pth"))

# Final test evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_test_accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {final_test_accuracy:.2f}% (best val acc was {best_val_accuracy:.2f}%)")
