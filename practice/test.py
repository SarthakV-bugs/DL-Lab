import torch
import torchvision.models as models

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Move model to device
model = model.to(device)

# Verify weights loaded (first conv layer)
with torch.no_grad():
    mean_val = model.conv1.weight.mean().item()
    std_val = model.conv1.weight.std().item()

if abs(mean_val) > 1e-5 or std_val > 1e-5:
    print(f"✅ ResNet-18 weights loaded successfully (mean={mean_val:.5f}, std={std_val:.5f})")
else:
    print("⚠️ Warning: weights may not be loaded properly!")

# Optional: print architecture summary
print(model)
