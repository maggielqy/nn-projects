import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)



class MyDataset(Dataset):
    def __init__(self,features,labels):
        self.X = torch.tensor(features,dtype=torch.float32)
        self.y = torch.tensor(labels,dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]


train_data = pd.read_csv('data/Consumo_cerveja_train.csv')
dev_data = pd.read_csv('data/Consumo_cerveja_dev.csv')

X_train = train_data.drop(columns = 'Consumo de cerveja (litros)', axis = 1).values
y_train = train_data['Consumo de cerveja (litros)'].values

X_dev = dev_data.drop(columns = 'Consumo de cerveja (litros)', axis = 1).values
y_dev = dev_data['Consumo de cerveja (litros)'].values

train_ds = MyDataset(X_train,y_train)
dev_ds = MyDataset(X_dev,y_dev)

train_dl = DataLoader(train_ds, batch_size = 16)
dev_dl = DataLoader(dev_ds, batch_size = dev_ds.__len__())



class MyNetwork(nn.Module):
    def __init__(self, lr):
        #inherit the functionalities from the superclass
        super(MyNetwork, self).__init__()
        #store lr to self
        self.lr = lr

        #input to hidden layer
        self.linear1 = nn.Linear(5,32)
        self.relu = nn.ReLU()
        #hidder to output layer
        self.linear2 = nn.Linear(32,1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def optimize(self, dataTrain, dataDev, epochs=20):
        
        lossFun = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr = self.lr)
        


        for epoch in range(epochs):
            self.train()

            #loop over each batch in dataTrain, calculate the loss and tune the parameters
            for batch in dataTrain:
                optimizer.zero_grad()
                #make predictions
                yhat = self.forward(batch[0])
                #compute loss
                #view(-1,1) turn yhat into the shape of (batchsize)
                loss = lossFun(yhat.view(-1), batch[1])
                #train the model
                loss.backward()
                optimizer.step()

            #print the mse for dataDev
            mse = 0
            with torch.no_grad():
                for b in dataDev:
                    y_p = self.forward(b[0])
                    loss_dev = lossFun(y_p.view(-1),b[1])
                    #use item to get a float value of the loss tensor
                    mse += loss_dev.item()
            
            print(f'Epoch {epoch+1}, the dev Mse is: {mse}')
            
        return mse


model = MyNetwork(lr=1e-3)

model.optimize(train_dl,dev_dl,20)


X_all = np.concatenate((X_train,X_dev),axis=0)
y_all = np.concatenate((y_train,y_dev),axis=0)

kf = sklearn.model_selection.KFold(n_splits=8,random_state=42, shuffle=True)


for i, (train_index, test_index) in enumerate(kf.split(X_all)):
    train_kfX = X_all[train_index]
    dev_kfX = X_all[test_index]

    train_kfy = y_all[train_index]
    dev_kfy = y_all[test_index]

    train_kfDS = MyDataset(train_kfX,train_kfy)
    dev_kfDS = MyDataset(dev_kfX,dev_kfy)

    train_dfDL = DataLoader(train_kfDS, batch_size=16)
    dev_dfDL = DataLoader(dev_kfDS, batch_size=dev_kfDS.__len__())

    model = MyNetwork(lr=1e-3)

    model.optimize(train_dfDL,dev_dfDL,20)


# model = MyNetwork(lr=1e-10)
# model.optimize(train_dl,dev_dl,20)

# model = MyNetwork(lr=1e-1)
# model.optimize(train_dl,dev_dl,20)

# model = MyNetwork(lr=1)
# mse = model.optimize(train_dl,dev_dl,20)
# print(f'mse is {mse}')

# model = MyNetwork(lr=10)
# model.optimize(train_dl,dev_dl,20)

lr_list = []
mse_list = []

for i in np.logspace(-10,1,12):
    lr_list.append(i)

    model = MyNetwork(lr=i)
    mse = model.optimize(train_dl,dev_dl,20)
    mse_list.append(round(mse,2))

mse_list

plt.plot(lr_list,mse_list)
plt.xscale('log')
plt.show()

#hypothesis that the best learning rating is 1e-1 is right
