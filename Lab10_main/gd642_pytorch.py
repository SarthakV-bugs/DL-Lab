import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

## Load CSV files
landmark_gene = pd.read_csv("1000G_landmark_genes.csv")
landmark_gene = landmark_gene.iloc[:, 4:].to_numpy().T  # Drop metadata columns
print("Landmark shape:", landmark_gene.shape)

target_gene = pd.read_csv("1000G_target_genes.csv")
target_gene = target_gene.iloc[:, 4:].to_numpy().T
print("Target shape:", target_gene.shape)
    
## Custom Dataset
class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = GeneExpressionDataset(landmark_gene, target_gene)



## Split into train, val and test
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

##get indices
indices = torch.randperm(total_size).tolist()
train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

x_train, y_train = landmark_gene[train_idx], target_gene[train_idx]
x_val, y_val = landmark_gene[val_idx], target_gene[val_idx]
x_test, y_test = landmark_gene[test_idx], target_gene[test_idx]


##Standardize using training set only to prevent leaking of val and test into the dataset
sc_X = StandardScaler().fit(x_train)
sc_y = StandardScaler().fit(y_train)

x_train = sc_X.transform(x_train)
x_val   = sc_X.transform(x_val)
x_test  = sc_X.transform(x_test)

y_train = sc_y.transform(y_train)
y_val   = sc_y.transform(y_val)
y_test  = sc_y.transform(y_test)



##load the datasets
train_set = GeneExpressionDataset(x_train, y_train)
val_set = GeneExpressionDataset(x_val, y_val)
test_set = GeneExpressionDataset(x_test, y_test)

##loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


print(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

## Model definition with BatchNorm
class GeneExpressNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256]):
        super(GeneExpressNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_sizes[1], output_size)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate model, loss, optimizer
model = GeneExpressNet(
    input_size=landmark_gene.shape[1],
    output_size=target_gene.shape[1]
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Training loop with loss tracking
train_losses = []
val_losses = []

for epoch in range(300):
    # Training
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    # Store losses
    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

## Testing
model.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()
print(f"\nFinal Test Loss: {test_loss/len(test_loader):.4f}")

# Example predictions
preds = outputs.detach().cpu().numpy()
true_vals = y_batch.detach().cpu().numpy()
print("Sample predictions vs actuals:")
for p, t in zip(preds[:5], true_vals[:5]):
    print(f"Predicted: {p}, Actual: {t}")



## Plot loss curves
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss Over Time")
plt.legend()
plt.grid(True)
plt.show()
