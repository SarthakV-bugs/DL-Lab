import torch
from torch import nn
from embeddings import vocab_size, caption_embedder, vocab, padded_caption, numerical_captions
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pickle

## Load the features
print("Current working directory:", os.getcwd())

# Load image features and detach to avoid gradient issues
raw_img_features = torch.load("/home/ibab/PycharmProjects/DL-Lab/Lab15/flickr8k_features.pth")
img_features = {k: v.detach().requires_grad_(False) for k, v in raw_img_features.items()}
print(type(img_features))  # dict

# Load captions embeddings and detach
captions_emb = torch.load("/home/ibab/PycharmProjects/DL-Lab/captions_embeddings.pt").detach().requires_grad_(False)

captions_idx = padded_caption.detach().requires_grad_(False)

# Build mapping from caption file
caption_file = "/home/ibab/PycharmProjects/DL-Lab/Lab15/flickr8k/captions.txt"
img_dir = "/home/ibab/PycharmProjects/DL-Lab/Lab15/flickr8k/Images"

mapping = []
with open(caption_file, 'r') as f:
    next(f)  # skip header if any
    for idx, line in enumerate(f):
        img_name, _ = line.strip().split(",", 1)
        img_path = str(Path(img_dir) / img_name)
        mapping.append((img_path, idx))


## Dataset class
class Flickr8kEmbeddedDataset(Dataset):
    def __init__(self, img_features, captions_emb, captions_idx, mapping):
        self.img_features = img_features  # dict of {img_path: tensor(256)}
        self.captions_emb = captions_emb  # padded, embedded captions (N, seq_len, 256)
        self.captions_idx = captions_idx  # padded numerical captions (N, seq_len)
        self.mapping = mapping  # [(img_path, idx), ...]

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        img_path, cap_idx = self.mapping[idx]

        img_feat = self.img_features[img_path].detach().clone()
        if img_feat.dim() == 1:
            img_feat = img_feat.unsqueeze(0)  # (1, 256)

        caption_emb = self.captions_emb[cap_idx].detach().clone()
        if caption_emb.dim() == 2:
            caption_emb = caption_emb.unsqueeze(0)  # (1, seq_len, 256)

        caption_idx = self.captions_idx[cap_idx].detach().clone()

        return img_feat, caption_emb, caption_idx


## Model definition
class CaptionGeneration(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super().__init__()

        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.img_fc = nn.Linear(embed_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.tanh = nn.Tanh()

    def forward(self, img_features, caption):
        # img_features: (batch_size, embed_size)
        # caption: (batch_size, seq_len, embed_size)

        h0 = self.tanh(self.img_fc(img_features)).unsqueeze(0)  # (num_layers, batch, hidden_size)
        output, hidden = self.rnn(caption, h0)
        logits = self.fc_out(output)  # (batch, seq_len, vocab_size)
        return logits


## Hyperparameters
embed_size = 256
hidden_size = 512
num_layers = 1
lr = 1e-3
num_epochs = 20
batch_size = 32

vocab_size = len(vocab)

dataset = Flickr8kEmbeddedDataset(img_features, captions_emb, captions_idx, mapping)

# Remove pin_memory=True since no GPU, but keep num_workers>0 for speed if possible
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=False)

model = CaptionGeneration(embed_size, vocab_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

save_dir = "Lab15/checkpoints"
os.makedirs(save_dir, exist_ok=True)

def train():
    for epoch in range(num_epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for img_feat, caption_emb, caption_idx in loop:
            optimizer.zero_grad()

            img_feat = img_feat.squeeze(1)  # (batch, 256)
            caption_emb = caption_emb.squeeze(1)  # (batch, seq_len, 256)

            logits = model(img_feat, caption_emb)

            # Shift targets for next word prediction
            logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            targets = caption_idx[:, 1:].contiguous().view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f"{save_dir}/caption_model_epoch_{epoch + 1}.pth")

    print(f"Model saved: {save_dir}/caption_model_epoch_{epoch + 1}.pth")

def main():
    train()

    # Save caption_embedder and vocab AFTER training
    torch.save(caption_embedder.state_dict(), f"{save_dir}/caption_embedder.pth")
    with open(f"{save_dir}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    main()
