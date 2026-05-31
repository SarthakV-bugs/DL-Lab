##using flickr8k dataset to create a custom dataset
##flick8k dataset has an image folder and a caption.txt file
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import models, transforms

##function to load the text file
def load_txt_annotations(txt_file):
    # loading the .txt file in cases without headers
    file_path = '/home/ibab/Desktop/DL_datasets/flickr8k/captions.txt'
    filenames = []
    captions = []

    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')

    if "image" in lines[0].lower() and "caption" in lines[0].lower():
        lines = lines[1:]

    for line in lines:
        filename, caption = line.split(",", 1)
        filenames.append(filename)
        captions.append(caption)

    return filenames, captions


#build a custom dataset
##when the file format = csv
# class flick8k_dataset(Dataset):#subclass from Dataset
#     def __init__(self, img_dir, csv_file, transform=None, target_transform=None):
#         """
#
#         :param img_dir: path to images directory
#         :param csv_file: path to csv file
#         :param transform: transform to apply to image
#         :param target_transform: transform to apply to label(optional)
#         """
#         #load the metadata , not the images
#         self.img_dir = img_dir
#         self.csv_file = pd.read_csv(csv_file) #works when header is present
#         self.transform = transform
#
#     def __len__(self):
#         #return the total number of samples
#         return len(self.csv_file)
#
#     def __getitem__(self, idx):
#         #retrieve the specific sample at 'index'
#
#         #get the image path from the csv file (column 0 is filename)
#         img_path = os.path.join(self.img_dir, self.csv_file.iloc[idx, 0]) #row, column
#
#         #open the image
#         image = Image.open(img_path).convert('RGB')
#
#         #get the label (col 1 is label)
#         y_label = self.csv_file.iloc[idx, 1]
#
#         #apply the transformations
#         if self.transform:
#             image = self.transform(image)
#
#         return image, y_label
#


class flick8k_dataset(Dataset):
    def __init__(self, img_dir,csv_file,transform=None):
        self.img_path = img_dir
        self.transform = transform

        #load only the filename and captions from .txt file
        self.filenames , self.captions = load_txt_annotations(csv_file)


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        img_name = self.filenames[idx]
        caption = self.captions[idx]

        img_path = os.path.join(self.img_path, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption


##hyperparameters
embed_dim = 256

##extract the features from the image for image captioning method
##use a pretrained model like ResNET
extractor = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')

#freeze the pretrained layers
for param in extractor.parameters():
    param.requires_grad = False

#input dimension of the final layers in ResNET
feature_dim = extractor.fc.in_features

#replace the classification head to get the 512 feature vector
extractor.fc = nn.Identity()

#project the 512 dimension to 256 dimension
img_proj = nn.Linear(feature_dim, embed_dim)

#evaluation model
extractor.eval()

print(f"ResNet18 feature dimension: {feature_dim}")
print(f"Projection: {feature_dim} -> {embed_dim}")



##Define the RNN model architecture 




dataset = flick8k_dataset(
    csv_file='/home/ibab/Desktop/DL_datasets/flickr8k/captions.txt',
    img_dir= '/home/ibab/Desktop/DL_datasets/flickr8k/Images')

#split the dataset into train data and test data
#use the random split method or subset method

#random split method
n = len(dataset)
train_size = int(0.8 * n)
test_size = n - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])
print(len(train_set))

#use dataloader to load batches lazily
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

#
# #subsetting method
# indices = list(range(n))
# train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
# train_sampler = SubsetRandomSampler(train_indices)
# test_sampler = SubsetRandomSampler(test_indices)




