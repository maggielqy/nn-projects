# imports
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

# load train and test set
fashion_trainset = torchvision.datasets.FashionMNIST('data/', train=True, download=True, transform=transforms.ToTensor())
fashion_testset = torchvision.datasets.FashionMNIST('data/', train=False, download=True, transform=transforms.ToTensor())

# get train and test loader
fashion_train_loader = torch.utils.data.DataLoader(dataset=fashion_trainset, batch_size=64, shuffle=True)
fashion_test_loader = torch.utils.data.DataLoader(dataset=fashion_testset, batch_size=64, shuffle=False)


# to store accuracy and loss after each epoch of training
accuracy_values = []
loss_values = []

# define  model
class BaseNetwork(nn.Module):
    def __init__(self, lr):
        #inherit the functionalities from the superclass
        super(BaseNetwork, self).__init__()
        #store lr to self
        self.lr = lr

        #input to hidden layer
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28,64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64,32)
        #hidder to output layer
        self.linear3 = nn.Linear(32,10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

    def optimize(self, dataTrain, dataDev, epochs=5):
        
        lossFun = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr = self.lr)

        for epoch in range(epochs):
            self.train()

            #loop over each batch in dataTrain, calculate the loss and tune the parameters
            for batch in dataTrain:
                optimizer.zero_grad()
                #make predictions
                yhat = self.forward(batch[0])
                #compute loss
                loss = lossFun(yhat, batch[1])
                #train the model
                loss.backward()
                optimizer.step()

            total_correct = 0
            total_samples = 0
            total_loss = 0
            total_batches = 0
            

            self.eval()

            with torch.no_grad():
                for b in dataDev:
                    y_p = self.forward(b[0])
                    loss_dev = lossFun(y_p,b[1])
                    #use item to get a float value of the loss tensor
                    total_loss += loss_dev.item()
                    total_batches += 1
                    predictions = y_p.argmax(dim=1)
                    total_correct += (predictions == b[1]).sum().item()
                    total_samples += b[1].size(0)

            cur_acc=total_correct / total_samples
            accuracy_values.append(cur_acc)

            cur_los = total_loss/total_batches
            loss_values.append(cur_los)

            print(f'Epoch {epoch}, the dev cross entropy loss is: {cur_los}')
            print(f'Epoch {epoch}, the accuracy is: {cur_acc}')
            
        return cur_acc, cur_los


accuracy_values = []
loss_values = []

model = BaseNetwork(lr=1e-3)
model.optimize(fashion_train_loader,fashion_test_loader,5)

# create  network class WITH batch normalisation
accuracy_values_bn = []
loss_values_bn = []

class BatchNormNetwork(nn.Module):
    def __init__(self, lr):
        #inherit the functionalities from the superclass
        super(BatchNormNetwork, self).__init__()
        #store lr to self
        self.lr = lr

        #input to hidden layer
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28,64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64,32)
        self.bn2 = nn.BatchNorm1d(32)
        #hidder to output layer
        self.linear3 = nn.Linear(32,10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

    def optimize(self, dataTrain, dataDev, epochs=5):
        
        lossFun = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr = self.lr)

        for epoch in range(epochs):
            self.train()

            #loop over each batch in dataTrain, calculate the loss and tune the parameters
            for batch in dataTrain:
                optimizer.zero_grad()
                #make predictions
                yhat = self.forward(batch[0])
                #compute loss
                loss = lossFun(yhat, batch[1])
                #train the model
                loss.backward()
                optimizer.step()

            total_correct = 0
            total_samples = 0
            total_loss = 0
            total_batches = 0
            

            self.eval()

            with torch.no_grad():
                for b in dataDev:
                    y_p = self.forward(b[0])
                    loss_dev = lossFun(y_p,b[1])
                    #use item to get a float value of the loss tensor
                    total_loss += loss_dev.item()
                    total_batches += 1
                    predictions = y_p.argmax(dim=1)
                    total_correct += (predictions == b[1]).sum().item()
                    total_samples += b[1].size(0)

            cur_acc=total_correct / total_samples
            accuracy_values_bn.append(cur_acc)

            cur_los = total_loss/total_batches
            loss_values_bn.append(cur_los)

            print(f'Epoch {epoch}, the dev cross entropy loss is: {cur_los}')
            print(f'Epoch {epoch}, the accuracy is: {cur_acc}')
            
        return cur_acc, cur_los

# instantiate and optimise  model with batch normalisation
accuracy_values_bn = []
loss_values_bn = []

model = BatchNormNetwork(lr=1e-3)
model.optimize(fashion_train_loader,fashion_test_loader,5)

print(accuracy_values)
print(loss_values)
print(accuracy_values_bn)
print(loss_values_bn)

# plot test accuracies for the baseline model and the model with batch normalisation
import matplotlib.pyplot as plt
plt.plot(range(1,6), accuracy_values, 's--', label="baseline")
plt.plot(range(1,6), accuracy_values_bn, 'o--', label='batch normalized')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy per Epoch')
plt.legend()
plt.show()

plt.plot(range(1,6), loss_values, 's--', label="baseline")
plt.plot(range(1,6), loss_values_bn, 'o--', label='batch normalized')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss per Epoch')
plt.legend()
plt.show()

accuracy_values_bn1 = []
loss_values_bn1 = []

class BatchNormNetwork1(nn.Module):
    def __init__(self, lr):
        #inherit the functionalities from the superclass
        super(BatchNormNetwork1, self).__init__()
        #store lr to self
        self.lr = lr

        #input to hidden layer
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28,64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64,32)
        self.bn2 = nn.BatchNorm1d(32)
        #hidder to output layer
        self.linear3 = nn.Linear(32,10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        #batch norm layer after activation
        x = self.bn1(x)
        x = self.linear2(x)
        x = self.relu(x)
        # batch norm layer after activation
        x = self.bn2(x)
        x = self.linear3(x)
        return x

    def optimize(self, dataTrain, dataDev, epochs=5):
        
        lossFun = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr = self.lr)

        for epoch in range(epochs):
            self.train()

            #loop over each batch in dataTrain, calculate the loss and tune the parameters
            for batch in dataTrain:
                optimizer.zero_grad()
                #make predictions
                yhat = self.forward(batch[0])
                #compute loss
                loss = lossFun(yhat, batch[1])
                #train the model
                loss.backward()
                optimizer.step()

            total_correct = 0
            total_samples = 0
            total_loss = 0
            total_batches = 0
            

            self.eval()

            with torch.no_grad():
                for b in dataDev:
                    y_p = self.forward(b[0])
                    loss_dev = lossFun(y_p,b[1])
                    #use item to get a float value of the loss tensor
                    total_loss += loss_dev.item()
                    total_batches += 1
                    predictions = y_p.argmax(dim=1)
                    total_correct += (predictions == b[1]).sum().item()
                    total_samples += b[1].size(0)

            cur_acc=total_correct / total_samples
            accuracy_values_bn1.append(cur_acc)

            cur_los = total_loss/total_batches
            loss_values_bn1.append(cur_los)

            print(f'Epoch {epoch}, the dev cross entropy loss is: {cur_los}')
            print(f'Epoch {epoch}, the accuracy is: {cur_acc}')
            
        return cur_acc, cur_los

accuracy_values_bn1 = []
loss_values_bn1 = []

model = BatchNormNetwork1(lr=1e-3)
model.optimize(fashion_train_loader,fashion_test_loader,5)

plt.plot(range(1,6), accuracy_values, 's--', label="baseline")
plt.plot(range(1,6), accuracy_values_bn, 'o--', label='BN before activation')
plt.plot(range(1,6), accuracy_values_bn1, '^--', label='BN after activation')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy per Epoch')
plt.legend()
plt.show()

plt.plot(range(1,6), loss_values, 's--', label="baseline")
plt.plot(range(1,6), loss_values_bn, 'o--', label='BN before activation')
plt.plot(range(1,6), loss_values_bn1, '^--', label='BN after activation')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss per Epoch')
plt.legend()
plt.show()
