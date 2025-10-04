import torch
import matplotlib.pyplot as plt
# from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import  DataLoader
from torch import nn


#loading the dataset
training = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

testing = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# for img,label in enumerate(training):
#     print(img,label)
#
# #iterating and visualizing the dataset
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training), size=(1,)).item()
#     img, label =(training[sample_idx])
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


#using dataloader to iterate through the dataset in batches
#created iterables

training_dataloader = DataLoader(training,batch_size=64, shuffle=True)
testing_dataloader = DataLoader(testing, batch_size=64, shuffle=True)

# for img, label in training_dataloader:
#     print(f"Image sample:{img}")
#     print(f"Label :{label}")

#Display image and label.
train_feature, train_label = next(iter(training_dataloader))
print(train_feature.size())
print(train_label.size())

print(train_feature[0].squeeze())
plt.imshow(train_feature[0].squeeze(),cmap="gray")
plt.show()


#define a neural net architecture here
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
logits = model(train_feature)
softmax = nn.Softmax(dim=1)
pred_prob=softmax(logits)
y_pred = pred_prob.argmax(1)
print(f"Predicted class: {y_pred}")
print(logits.size())


print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")