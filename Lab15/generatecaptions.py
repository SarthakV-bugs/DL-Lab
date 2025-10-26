import matplotlib.pyplot as plt
from PIL import Image
import torch
from embeddings import vocab, caption_embedder
from vanilla_rnn import CaptionGeneration

# Load model
model = CaptionGeneration(256, len(vocab), 512, 1)
checkpoint = torch.load("Lab15/checkpoints/caption_model_epoch_20.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def generate_caption(img_feat, max_len=20):
    with torch.no_grad():
        device = next(model.parameters()).device
        img_feat = img_feat.to(device)

        # Initial hidden state from image features
        h = model.tanh(model.img_fc(img_feat.unsqueeze(0))).unsqueeze(0)  # (1, 1, hidden_size)

        # Start token <SOS>
        input_idx = torch.tensor([[vocab.stoi["<SOS>"]]], device=device)
        input_token = caption_embedder(input_idx)  # embedding (1,1,embed_size)

        caption_idx = []

        for _ in range(max_len):
            out, h = model.rnn(input_token, h)  # out: (1,1,hidden_size)
            logits = model.fc_out(out.squeeze(1))  # (1, vocab_size)
            next_token = torch.argmax(logits, dim=-1).item()
            if next_token == vocab.stoi["<EOS>"]:
                break
            caption_idx.append(next_token)
            input_idx = torch.tensor([[next_token]], device=device)
            input_token = caption_embedder(input_idx)

        words = [vocab.itos[idx] for idx in caption_idx]
        return " ".join(words)


def visualize_image_caption(img_path, img_features):
    caption = generate_caption(img_features)
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()


# Example usage
img_path = "/home/ibab/PycharmProjects/DL-Lab/Lab15/flickr8k/Images/380034515_4fbdfa6b26.jpg"  # Replace with a valid image path
img_features_dict = torch.load("/home/ibab/PycharmProjects/DL-Lab/Lab15/flickr8k_features.pth")
img_features = img_features_dict[img_path]

visualize_image_caption(img_path, img_features)
