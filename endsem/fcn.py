import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm

# --- Hyperparams / device
batch_size = 64
lr = 1e-2
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data
transform = transforms.ToTensor()
train_ds = datasets.MNIST(root="~/Desktop/DL_datasets", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="~/Desktop/DL_datasets", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# --- Model matching assignment:
# Input: 784, Hidden1: 128 ReLU, Hidden2: 64 ReLU, Output: 10 (softmax for probs if needed)
class FCN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)   # logits -> CrossEntropyLoss expects logits

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits

model = FCN_MNIST().to(device)
opt = torch.optim.SGD(model.parameters(), lr=lr)
crit = nn.CrossEntropyLoss()

# --- Train for num_epochs
for epoch in range(1, num_epochs+1):
    model.train()
    train_correct = 0
    train_total = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        out = model(inputs)
        loss = crit(out, labels)
        loss.backward()
        opt.step()

        preds = out.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        loop.set_postfix(loss=loss.item())

    train_acc = 100.0 * train_correct / train_total

    # --- Evaluate on test set (once per epoch)
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            preds = out.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100.0 * test_correct / test_total
    print(f"Epoch {epoch}: Train acc: {train_acc:.2f}%  |  Test acc: {test_acc:.2f}%")

print("Training finished.")



