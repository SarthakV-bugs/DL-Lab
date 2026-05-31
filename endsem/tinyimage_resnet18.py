from torch.utils.data import Dataset
from PIL import Image
import os

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_path):
                img_path = os.path.join(cls_path, fname)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

import torch
from torch import nn
from torchvision import models

def get_pretrained_model(num_classes):
    model = models.resnet18(pretrained=True)

    # OPTIONAL: freeze feature extractor (good for small datasets)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

from torchvision import transforms

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

from torch.utils.data import DataLoader, random_split

TRAIN_DIR = "/path/train"
TEST_DIR  = "/path/test"

full_train_dataset = TinyImageNetDataset(TRAIN_DIR, transform=transform_train)

# Split TRAIN into TRAIN + VAL
val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

test_dataset = TinyImageNetDataset(TEST_DIR, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct, total, running_loss = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, running_loss = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100 * correct / total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(full_train_dataset.classes)

model = get_pretrained_model(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

best_val_acc = 0
best_model_path = "best_resnet18.pth"

EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
    print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%")

    # save best model (based only on VAL)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print("  Saved new best model.")


# Load best model before testing
# model.load_state_dict(torch.load(best_model_path))
# model.to(device)

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nFINAL TEST ACCURACY: {test_acc:.2f}%")
