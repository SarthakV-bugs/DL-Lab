#Train CIFAR-10 classifier using pre-trained ResNet-18
#Extract feature vector + linear layer

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch import nn

#transform block
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

#load the dataset

train_data = datasets.CIFAR10(
    root= "data",
    train=True,
    download=True,
    transform= transform

)


test_data = datasets.CIFAR10(
    root= "data",
    train=False,
    download=True,
    transform= transform

)

#create dataloaders for the train and test set

train_loader = DataLoader(train_data,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(test_data,
                          batch_size=64,
                          shuffle=False)


#use RESnet18 model with pre-trained weights for feature extraction
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

#freeze the pre-trained layers
for param in model.parameters():
    print(param.size())
    param.requires_grad = False #no gradient update on the hidden layer

#initialize the new output layer for 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # Only optimize the final layer



def train(train_loader, model, loss_fn, optimizer):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for batch, (x, y) in enumerate(train_loader):

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = pred.argmax(dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    print(f"Training Loss: {avg_loss:.4f} | Training Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


#testing /evaluation loop
def test(test_loader,model, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch, (x,y) in enumerate(test_loader):

            #compute prediciton
            pred = model(x)
            # print(pred.shape)

            #compute loss
            loss = loss_fn(pred,y)
            test_loss += loss.item()

            #get predicted class for calculating the accuracy
            predicted = pred.argmax(dim=1)

            #count of correct prediction
            correct += (predicted == y).sum().item()

        avg_loss = test_loss/ len(test_loader) #avg loss across the batch size
        accuracy = correct / len(test_data)

    print(f"Test_loss: {avg_loss:.4f} | test_accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


num_epoch = 20
for t in range(num_epoch):
    print(f"Epoch: {t+1}")
    print(f"Training starts here:")
    train_loss, train_acc = train(train_loader, model, loss_fn, optimizer)
    print(f"Testing starts here:")
    test_loss, test_acc = test(test_loader, model, loss_fn)
print("Done")

