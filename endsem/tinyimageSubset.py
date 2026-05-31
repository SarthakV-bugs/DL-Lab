##Implement a CNN model for image classification using TinyImageNet dataset
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os

from torchvision import transforms

from TransferLearning.FE2 import running_loss


##directly use the below method if the dir has the structure of train and test subfolders
# from torchvision.datasets import ImageFolder
#
# train_dir = "/home/ibab/Desktop/DL_datasets/imagenet_subset/train"
# test_dir = "/home/ibab/Desktop/DL_datasets/imagenet_subset/test"
#
# train_dataset = ImageFolder(train_dir)
# test_dataset = ImageFolder(test_dir)
#
# print(train_dataset)
# print(test_dataset)

class TinyIMAGENET(Dataset):
    def __init__(self,  root_dir, transform=None):
        
        self.root_dir = root_dir #path to the tiny image dataset
        self.transform = transform

        #get class folders
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls:i for i, cls in enumerate(self.classes)} #convert the classnames into numbers(ids)


        #collect (img_path, label) list of all the images
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            for filename in os.listdir(cls_path):
                img_path = os.path.join(cls_path, filename)
                if os.path.isfile(img_path) :
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label




class Tiny_CNN(nn.Module):
    def __init__(self, num_classes: int):
        super(Tiny_CNN, self).__init__()
    

        #CONV blocks
        self.features = nn.Sequential(
            #conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), 
            
            #conv2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #conv3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        
        )

        #classifier block
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



##training, evaluation

def train_one_epoch(model, optimizer, criterion, data_loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images
        labels = labels
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(labels.data).sum().item()
        total += labels.size(0)
        
        if batch_idx % 100 == 0:
            avg_loss = running_loss / total
            accuracy = 100 * correct / total
            
    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy


def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images
            labels = labels
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum().item()
            total += labels.size(0)
            
    loss = running_loss / total
    accuracy = 100 * correct / total
    return loss, accuracy

    
    
    
def main():
    
    #configs
    
    train_dir = "/home/ibab/Desktop/DL_datasets/imagenet_subset/train"
    test_dir = "/home/ibab/Desktop/DL_datasets/imagenet_subset/test"
    image_size = 224
    num_epoch = 10
    batch_size = 64
    learning_rate = 0.001


    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    #dataset
    train_dataset = TinyIMAGENET(
        root_dir=train_dir,
        transform=transform
    )
    
    test_dataset = TinyIMAGENET(
        root_dir= test_dir,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Tiny_CNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #training loop
    for epoch in range(num_epoch):
        loss, accuracy = train_one_epoch(model, optimizer, criterion, train_loader)
        print("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch, loss, accuracy))
        #
        # #evaluate on test set each epoch
        # test_loss, test_accuracy = test(model, test_loader)
        # print("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(test_loss, test_accuracy))
        #
    final_loss, final_accuracy = test(model, test_loader)
    print("Final loss: {:.4f}, Final accuracy: {:.4f}".format(final_loss, final_accuracy))

if __name__ == "__main__":
    main()