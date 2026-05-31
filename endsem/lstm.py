# #write a python function to generate a dataset of 400 DNA sequences, "ACGTAGCTAGCT",
# #"GCTAGCTAGGCA", "CGTACGTAGCTA", "TGCATGCATGCA", have labels as 0,1,0,1
# #repeat them 100 times
# from sympy.series import sequences
# from torch.utils.data import Dataset
#
#
# #a)
# def generate_dnaseq(sequences):
#     seq_dataset = []
#     for i in range(100):
#         for key, value in sequences.items():
#             seq_dataset.append((key, sequences[key]))
#     return seq_dataset
#
# #b)lstm classifier
#
# class DNAsequenceDataset(Dataset):
#     """
#     Expects a list of (sequence, label) pairs.
#     Encodes A,C,G,T -> 0,1,2,3 and returns:
#     """
#     #define mapping for A,C,G,T
#     mapping = {
#         "A":0, "C":1, "G":2, "T":3
#     }
#
#     def __init__(self, sequences):
#         self.sequences = sequences
#
#         #
#
#
#
#
#
#
# def main():
#     sequences = {"ACGTAGCTAGCT": 0,
#                  "GCTAGCTAGGCA": 1,
#                  "CGTACGTAGCTA": 0,
#                  "TGCATGCATGCA": 1}
#
#     #dataset
#     x = generate_dnaseq(sequences)
#
# if __name__ == '__main__':
#     main()
#
#
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

#(a) generate 400 samples
def generate_dnaseq():
    dna_sequences = ["ACGTAGCTAGCT", "GCTAGCTAGGCA", "CGTACGTAGCTA", "TGCATGCATGCA"]
    labels = [0, 1, 0, 1]
    data = []
    for seq, lab in zip(dna_sequences, labels):
        for _ in range(100):
            data.append((seq, lab))
    random.shuffle(data)
    sequences, labels = zip(*data)
    return list(sequences), np.array(labels, dtype=np.int64)

# ---------- Dataset ----------
class DNACharDataset(Dataset):
    # include 'N' in mapping so padding doesn't break
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    def __init__(self, sequences, labels):
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels = labels
        self.max_len = max(len(s) for s in sequences)

    def __len__(self):
        return len(self.labels)

    def encode_and_pad(self, seq: str):
        padded = seq + 'N' * (self.max_len - len(seq))
        return [self.char_to_int[ch] for ch in padded]

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(self.encode_and_pad(seq), dtype=torch.long)  # (seq_len,)
        y = torch.tensor(self.labels[idx], dtype=torch.float)         # scalar float for BCE
        return x, y

# ---------- Model ----------
class DNANet(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # vocab includes 'N'
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim // 2, 1)  # single logit output

    def forward(self, x):
        # x: (batch, seq_len) long
        emb = self.embedding(x)           # (batch, seq_len, embed_dim)
        out1, _ = self.lstm1(emb)         # (batch, seq_len, hidden)
        out2, _ = self.lstm2(out1)        # (batch, seq_len, hidden2)
        last = out2[:, -1, :]             # last time-step
        logit = self.fc(last).squeeze(1)  # (batch,)
        return logit                      # raw logits (no sigmoid)

# ---------- Training / evaluation ----------
def train_and_evaluate():
    # data
    sequences, labels = generate_dnaseq()   # 400 samples
    dataset = DNACharDataset(sequences, labels)

    # split into train/val
    val_frac = 0.2
    n = len(dataset)
    n_val = int(n * val_frac)
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNANet().to(device)

    # use BCEWithLogitsLoss (expects logits)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)                 # (batch,)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == y_batch.long()).sum().item()
            total += x_batch.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # validation
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                logits = model(x_val)
                loss_v = criterion(logits, y_val)
                val_loss_total += loss_v.item() * x_val.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == y_val.long()).sum().item()
                val_total += x_val.size(0)

        val_loss = val_loss_total / val_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch}/{epochs}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

    # final eval on validation (or a separate test set if you have one)
    print("Done.")

if __name__ == "__main__":
    train_and_evaluate()
