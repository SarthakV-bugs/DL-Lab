##Part A) Feature extraction from the Flickr8 dataset using pre-trained models

##Numerical representation for the images in flickr8k dataset 

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch import nn
import os

#embed size
embed_size = 256

#transform block
transform = transforms.Compose([
    transforms.Resize((224,224)),  #to match with the ImageNet pixel sizes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean
                        std=[0.229, 0.224, 0.225])      # ImageNet std
])

##create a custom dataset
class Flickr8kCustom(Dataset):
    def __init__(self, img_dir, caption_file, transform=None):

        self.img_dir = img_dir
        self.transform = transform

        #load the captions
        self.data = []
        with open(caption_file, 'r') as f:
            next(f)  # Skip the header line, consists of image, caption headings
            for line in f:
                img_name, caption = line.strip().split(',',1)
                img_path = os.path.join(img_dir, img_name)

                #add into the data if the image exists
                if os.path.exists(img_path):
                    self.data.append((img_path,caption))


        # Group by image for multiple captions
        self.grouped = {}
        for img_path, caption in self.data:
            if img_path not in self.grouped:
                self.grouped[img_path] = [] #first time seeing the img, initialize an empty list
            self.grouped[img_path].append(caption) #add the caption to this image list in the dict

        self.images = list(self.grouped.keys()) #keys comprises of the images and values correspond to the caption
        print(f"Loaded {len(self.images)} images with {len(self.data)} captions")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB') #opens the image

        if self.transform:
            image = self.transform(image)

        captions = self.grouped[img_path] #gets the captions associated with that image
        return image, captions


#load the complete dataset
#Default torch datasets throws path error
data = Flickr8kCustom(
    img_dir= "/home/ibab/PycharmProjects/DL-Lab/Lab15/flickr8k/Images", #path to image folder
    caption_file = "/home/ibab/PycharmProjects/DL-Lab/Lab15/flickr8k/captions.txt", #path to caption file
    transform= transform
)

#dataloader
dataloader = DataLoader(dataset=data, batch_size=32, shuffle=True)

#use RESnet18 model with pre-trained weights for feature extraction
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

#freeze the pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Get the input dimension of the final layer
# ResNet18's fc layer has 512 input features
feature_dim = model.fc.in_features  # This will be 512

# Replace the classification layer with Identity to get raw features
model.fc = nn.Identity()

# Create a separate projection layer to map features to embedding size
img_proj = nn.Linear(feature_dim, embed_size)  # 512 -> 256

# Set model to evaluation mode
model.eval()

print(f"ResNet18 feature dimension: {feature_dim}")
print(f"Projection: {feature_dim} -> {embed_size}")

features_dict = {}

#extraction loop
with torch.no_grad():
    for batch_idx, (images, captions) in enumerate(dataloader):
        # images shape: [batch_size, 3, 224, 224]
        features = model(images)  # [batch_size, 512]
        projected_features = img_proj(features)  # [batch_size, 256]

        # Store each image's features
        for i in range(len(images)):
            img_idx = batch_idx * dataloader.batch_size + i
            if img_idx < len(data):
                img_path = data.images[img_idx]
                features_dict[img_path] = projected_features[i]

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1} batches")

    torch.save(features_dict, 'flickr8k_features.pth')
    print(f"Features saved to 'flickr8k_features.pth'")










