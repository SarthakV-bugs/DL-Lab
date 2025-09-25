#Download MNIST dataset and implement a MNIST classifier using CNN PyTorch library.
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

#resolves the proxy issue
import os
os.environ['HTTP_PROXY']  = "http://245hsbd009%40ibab.ac.in:verma1903@proxy.ibab.ac.in:3128/"
os.environ['HTTPS_PROXY'] = "http://245hsbd009%40ibab.ac.in:verma1903@proxy.ibab.ac.in:3128/"


#tranform the dataset into tensors object
#explore transforms for pre-processing of the images
transform = transforms.ToTensor()

#load the MNIST dataset using torchvision lib
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train = True,
    transform=transform,
    download=True,
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train = False,
    transform=transform,
    download=True,
)
print(f"train_size = {len(train_dataset)}, test_size = {len(test_dataset)}") #sample sizes


#dataloader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1000,
                         shuffle=False) #1000/10000

# images = [] #60,000 train images
# labels = []
# for img, label in train_loader:
#     images.append(img)
#     labels.append(label)


#define CNN
#flow of the input
#conv1 -> ReLU -> POOL -> conv2 -> ReLU -> POOL -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> softmax
class MNISTNet(nn.Module):
    def __init__(self):
        """
        Network design
        """
        super().__init__() #super constructor

        self.conv1 = nn.Conv2d(1,6, 5) #outs size of 24*24*6
        self.pool = nn.MaxPool2d(kernel_size=3) #outs size of 8*8*6
        self.conv2 = nn.Conv2d(6, 16, 5)#outs size of 4*4*16
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*4*4, 120) #in size 256
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #10 classes

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x))) #passing the input in the first CNN block
        x = F.relu(self.conv2(x)) #no pooling here

        #before forwarding the input to the fc layers, flatten the input
        x = self.flatten(x)
        # print(f"flattened input: {x.shape}")
        #forward the flattened input into the fc layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) #output layer
        #softmax not needed because of the crossEntropy loss already does it internally
        return x

#instantiate the model
model = MNISTNet()

##model parameters printing
# param = list(model.parameters())
# for i,p in enumerate(param):
#     print(p)
# for name, param in model.named_parameters():
#     print(name, param.shape)


#define the loss
criterion = nn.CrossEntropyLoss()


#gradient descent over the batch size mentioned in the dataloader
#As my batch-size = 64, it will implement mini batch gradient descent
optimizer = torch.optim.SGD(model.parameters(),lr=0.001) #regardless of the mention .SGD

for epoch in range(100): #100 iterations
    running_loss = 0

    #accuracy check
    correct = 0 #correct predictions
    total = 0 #total predictions

    for img, label in train_loader:

        output = model(img)
        # print(f"fc3 Logits: {output}") #raw logits

        # comparing predicted vs actual labels
        # print(f"Softmax: {(F.softmax(output, dim=1))}") # prob values
        # print("Predicted class:", torch.argmax(F.softmax(output), dim=1))
        # print("Predicted class:", torch.max(F.softmax(output), dim=1))

        # print(f"label values: {label}") #label values

        loss = criterion(output,label)
        # print(f"loss: {loss.item()}")

        #define gradient update
        optimizer.zero_grad() #clears prev gradient
        loss.backward() #calculates the grad using autograd
        optimizer.step() #param updates , optimizer dependant

        #accuracy check
        running_loss += loss.item()
        pred = torch.argmax(output,dim=1)
        correct += (pred==label).sum().item() #correct predictions from the total predictions
        total += label.size(0) #batch size

    train_accuracy = 100 * correct/total
    avg_loss = running_loss/ len(train_loader)
#
# print(f"Accuracy of train: {train_accuracy:.2f} ")
# print(f"Average loss: {avg_loss:.2f}")


    #model evaluation
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad(): # no gradient tracking in eval
        for img, label in test_loader:
            output = model(img)
            loss = criterion(output, label)
            test_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            test_correct += (preds == label).sum().item()
            test_total += label.size(0)

    test_acc = 100 * test_correct / test_total
    test_loss = test_loss / len(test_loader)

    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_accuracy:.2f}% | Test Loss: {test_loss:.4f} | "
          f"Test Acc: {test_acc:.2f}%")










































# #inherit from the class dataset to create a custom dataset class
# class MNISTDataset(Dataset):
#
#     def __init__(self,data,labels):
#         self.data = data #sample data
#         self.labels = labels #labels for the dataset
#
#     def __len__(self):
#         return len(self.data) #returns the length of the sample data
#
#     def __getitem__(self, index):
#         """
#
#         :param index: idx of the sample
#         :return: returns sample and label at the input idx
#         """
#         sample = self.data[index] #retrieves the sample at the input index
#         labels = self.labels[index] #retrieves the corresponding label at that input
#         return sample,labels


