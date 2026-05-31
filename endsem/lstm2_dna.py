import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

# (a) Generate 400 DNA samples
def generate_dnaseq(sequences):
    seq_dataset = []

    # Repeat the 4 sequences 100 times => 400 samples
    for _ in range(100):
        for seq, label in sequences.items():
            seq_dataset.append((seq, label))

    # Separate X and Y
    x = [item[0] for item in seq_dataset]   # sequences
    y = [item[1] for item in seq_dataset]   # labels
    return x, y

# (b) Custom Dataset
class DNAsequenceDataset(Dataset):

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def encode(self, seq):
        return [self.mapping[ch] for ch in seq]

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoded = torch.tensor(self.encode(seq), dtype=torch.long)
        return encoded, torch.tensor(label, dtype=torch.float)


# (c) LSTM Model
class DNALSTM(nn.Module):
    def __init__(self , architecture='GRU'):
        super().__init__()

        self.embedding = nn.Embedding(4, 8)          # 4 letters → 8-dim embedding
        # self.lstm = nn.LSTM(8, 16, batch_first=True) # 8→16

        self.architecture = architecture

        #select layer based on architecture
        if architecture == 'LSTM':
            self.rnn_layer = nn.LSTM(8, 16, batch_first=True)
        elif architecture == 'GRU':
            self.rnn_layer = nn.GRU(8, 16, batch_first=True)
        else:
            self.rnn_layer = nn.RNN(8, 16, batch_first=True)

        self.fc = nn.Linear(16, 1)                   # binary output

    def forward(self, x):
        x = self.embedding(x)
        # RNN/GRU return: output, hidden
        # LSTM returns:   output, (hidden, cell)
        # We only care about 'output' which contains features for all timesteps
        out, _ = self.rnn_layer(x)
        final = out[:, -1, :]        # take last timestep
        return self.fc(final).squeeze()

# (d) Train + Test
def main():

    # original 4 sequences
    sequences = {
        "ACGTAGCTAGCT": 0,
        "GCTAGCTAGGCA": 1,
        "CGTACGTAGCTA": 0,
        "TGCATGCATGCA": 1
    }

    #Generate 400 samples
    X, Y = generate_dnaseq(sequences)

    #Build Dataset
    dataset = DNAsequenceDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #Model setup
    model = DNALSTM()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Train
    for epoch in range(10):
        total_loss = 0
        for seq, lbl in dataloader:

            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/10, Loss: {total_loss:.4f}")

    #Test Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for seq, lbl in dataloader:
            output = torch.sigmoid(model(seq))
            preds = (output >= 0.5).float()
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)

    print(f"Accuracy on 400 samples: {100 * correct / total:.2f}%")


if __name__ == '__main__':
    main()
