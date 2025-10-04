# Train CIFAR-10 classifier using pre-trained ResNet-18 + SVM

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch import nn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# transform block
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

# load the dataset
train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# create dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ResNet-18 and remove final layer
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


# Extract features
def extract_features(data_loader):
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)  # Flatten
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(features_list), np.concatenate(labels_list)


print("Extracting training features...")
train_features, train_labels = extract_features(train_loader)
print(f"Training features: {train_features.shape}")

print("Extracting test features...")
test_features, test_labels = extract_features(test_loader)
print(f"Test features: {test_features.shape}")

# Train SVM
print("\nTraining SVM...")
svm = LinearSVC(max_iter=1000, verbose=1)
svm.fit(train_features, train_labels)

# Evaluate
train_pred = svm.predict(train_features)
test_pred = svm.predict(test_features)

print(f"\nTrain Accuracy: {accuracy_score(train_labels, train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(test_labels, test_pred):.4f}")
