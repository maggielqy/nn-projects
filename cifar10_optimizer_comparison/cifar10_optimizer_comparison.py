
# library imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torcheval.metrics import MulticlassAccuracy

import numpy as np
from tqdm import tqdm

torch.manual_seed(42)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = trainset.classes



class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.stack(x)
        return outputs

    def optimize(self, train_loader, test_loader, lr=0.001, epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        metric = MulticlassAccuracy(num_classes=10)

        loss_hist=[]
        acc_hist=[]

        for epoch in range(epochs):
            self.train()
            loss_epoch=0

            for images,labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_epoch+=loss.item()

            #add average loss for epoch to history
            cur_loss = loss_epoch/len(train_loader)
            loss_hist.append(cur_loss)
                
            self.eval()
            metric.reset()
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = self.forward(images)
                    metric.update(outputs, labels)

            accuracy = metric.compute().item()
            acc_hist.append(accuracy)    

            print(f'Epoch: {epoch+1}, Loss: {cur_loss}, Accuracy: {accuracy}')
        
        return loss_hist, acc_hist




model = MyNetwork()
loss_sgd, acc_sgd = model.optimize(trainloader, testloader, lr=0.001, epochs=10)


# Set our seed for reproducibility
torch.manual_seed(42)

def get_optimizer(optimizer_name: str,
                  model: nn.Module,
                  learning_rate: float):
  """
  Function that returns an optimizer object that will be used
  for training

  Params:
  optimizer_name: This will indicate the type of optimizer to return
  model: Model which has the parameters to be optimized
  learning_rate: learning rate to be used
  """
  params = model.parameters()

  if optimizer_name == "momentum":
    return optim.SGD(params, lr=learning_rate, momentum=0.9)
  elif optimizer_name == "adagrad":
        return optim.Adagrad(params, lr=learning_rate)
  elif optimizer_name == "rmsprop":
    return optim.RMSprop(params, lr=learning_rate)
  elif optimizer_name == "adam":
    return optim.Adam(params, lr=learning_rate)
  elif optimizer_name == "sgd":
    return optim.SGD(params, lr=learning_rate)
  else:
    raise NotImplementedError


class MyNetwork2(nn.Module):
    def __init__(self):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.stack(x)
        return outputs

    def optimize(self, train_loader, test_loader, lr=0.001, epochs=10, optimizer_name=None):
        criterion = nn.CrossEntropyLoss()
        metric = MulticlassAccuracy(num_classes=10)
        optimizer = get_optimizer(optimizer_name, self, lr)

        loss_hist=[]
        acc_hist=[]

        for epoch in range(epochs):
            self.train()
            loss_epoch=0

            for images,labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_epoch+=loss.item()

            #add average loss for epoch to history
            cur_loss = loss_epoch/len(train_loader)
            loss_hist.append(cur_loss)
                
            self.eval()
            metric.reset()
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = self.forward(images)
                    metric.update(outputs, labels)

            accuracy = metric.compute().item()
            acc_hist.append(accuracy)    

            print(f'Epoch: {epoch+1}, Loss: {cur_loss}, Accuracy: {accuracy}')
        
        return loss_hist, acc_hist


model1 = MyNetwork2()
loss_mom, acc_mom = model1.optimize(trainloader, testloader, lr=0.001, epochs=5, optimizer_name='momentum')


model2 = MyNetwork2()
loss_ag, acc_ag = model2.optimize(trainloader, testloader, lr=0.001, epochs=5, optimizer_name='adagrad')


model3 = MyNetwork2()
loss_rms, acc_rms = model3.optimize(trainloader, testloader, lr=0.001, epochs=5, optimizer_name='rmsprop')


model4 = MyNetwork2()
loss_adam, acc_adam = model4.optimize(trainloader, testloader, lr=0.001, epochs=5, optimizer_name='adam')


import matplotlib.pyplot as plt

epochs = range(1, 6)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(epochs, loss_mom, 's--', label="SGD with momentum")
ax1.plot(epochs, loss_ag, 'x--', label='Adagrad')
ax1.plot(epochs, loss_rms, 'o--', label='RMSprop')
ax1.plot(epochs, loss_adam, '^--', label="Adam")
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Traning Loss')
ax1.set_title('Training Loss per Epoch')
ax1.legend()

ax2.plot(epochs, acc_mom, 's--', label="SGD with momentum")
ax2.plot(epochs, acc_ag, 'x--', label='Adagrad')
ax2.plot(epochs, acc_rms, 'o--', label='RMSprop')
ax2.plot(epochs, acc_adam, '^--', label="Adam")
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Test Accuracy per Epoch')
ax2.legend()

plt.show()


