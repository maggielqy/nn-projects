# imports
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

torch.manual_seed(42)

exam_data = pd.read_csv("Exam_Score_Prediction.csv")
exam_data['pass'] = (exam_data['exam_score'] >= 50).astype(int)
exam_data = exam_data.drop(["student_id", "exam_score"], axis = 1)
exam_data = pd.get_dummies(exam_data, dtype=int)



class ExamDataset(Dataset):
    def __init__(self,features,labels):
        self.X = torch.tensor(features,dtype=torch.float32)
        self.y = torch.tensor(labels,dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
    

training_data, validation_data = train_test_split(exam_data, train_size=0.8)
train_dataset = ExamDataset(training_data.drop(columns = 'pass', axis = 1).values, training_data['pass'].values)
validation_dataset = ExamDataset(validation_data.drop(columns="pass", axis = 1).values, validation_data['pass'].values)
train_dl = DataLoader(train_dataset, batch_size = 64)
validation_dl = DataLoader(validation_dataset, batch_size = validation_dataset.__len__())





class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(30,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,1)
        )

        #combined sigmoid and bceloss function. The documentation says this is more stable than a sigmoid + bceLoss function.
        #Is there any reason to apply them seperately?
        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, x):
        logits = self.layers(x)
        return logits
    
    def optimize(self, training_loader, validation_loader, epochs = 20, learning_rate = 0.0001):
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        for epoch in range(epochs):
            self.train()
            for batch in training_loader:
                optimizer.zero_grad()
                logits = self.forward(batch[0])
                loss_value = self.loss(logits.view(-1), batch[1])
                loss_value.backward()
                optimizer.step()

            total_correct = 0
            total_samples = 0
            total_loss = 0
            with torch.no_grad():
                for b in validation_loader:
                    logits = self.forward(b[0])
                    probabilities = torch.sigmoid(logits)
                    loss_dev = self.loss(logits.view(-1),b[1])
                    total_loss += loss_dev.item()
                    predictions = (probabilities > 0.5).int().view(-1)
                    total_correct = (predictions == b[1]).sum().item()
                    total_samples += b[1].size(0)
            print(f'Epoch {epoch}, the dev cross entropy loss is: {total_loss}')
            print(f'Epoch {epoch}, the accuracy is: {total_correct / total_samples}')




model = BaseNetwork()

model.optimize(train_dl, validation_dl)



class BaseNetworkWithEarlyStopping(nn.Module):

    def __init__(self):
        super(BaseNetworkWithEarlyStopping, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(30,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,1)
        )
        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, x):
        logits = self.layers(x)
        return logits
    
    def optimize(self, training_loader, validation_loader, patience, minimum_improvement, learning_rate = 0.0001):
        patience_count = 0
        epoch = 1
        previous_loss = 1
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        while patience_count <= patience:
            self.train()
            for batch in training_loader:
                optimizer.zero_grad()
                logits = self.forward(batch[0])
                loss_value = self.loss(logits.view(-1), batch[1])
                loss_value.backward()
                optimizer.step()

            total_correct = 0
            total_samples = 0
            total_loss = 0
            with torch.no_grad():
                for b in validation_loader:
                    logits = self.forward(b[0])
                    probabilities = torch.sigmoid(logits)
                    loss_dev = self.loss(logits.view(-1),b[1])
                    total_loss += loss_dev.item()
                    predictions = (probabilities > 0.5).int().view(-1)
                    total_correct = (predictions == b[1]).sum().item()
                    total_samples += b[1].size(0)
                    loss_improvement = previous_loss - total_loss
            print(f'Epoch {epoch}, the dev cross entropy loss is: {total_loss}')
            print(f'Epoch {epoch}, the dev cross entropy loss improvement is: {loss_improvement}')
            print(f'Epoch {epoch}, the accuracy is: {total_correct / total_samples}')
            epoch += 1
            if (loss_improvement < minimum_improvement):
                patience_count += 1
            else:
                patience_count = 0
            previous_loss = total_loss
            



model_early_stop = BaseNetworkWithEarlyStopping()
model_early_stop.optimize(train_dl, validation_dl, 3, 0.01)


class BaseNetworkWithL1(nn.Module):

    def __init__(self):
        super(BaseNetworkWithL1, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(30,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,1)
        )
        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, x):
        logits = self.layers(x)
        return logits
    
    def optimize(self, training_loader, validation_loader, l_one_weight, epochs = 20, learning_rate = 0.0001):
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        for epoch in range(epochs):
            self.train()
            for batch in training_loader:
                optimizer.zero_grad()
                logits = self.forward(batch[0])
                loss_value = self.loss(logits.view(-1), batch[1])
                l1_regularisation_weight = sum(param.abs().sum() for name, param in self.named_parameters() if "weight" in name)
                loss_value_regularised = loss_value + l_one_weight * l1_regularisation_weight
                loss_value_regularised.backward()
                optimizer.step()

            total_correct = 0
            total_samples = 0
            total_loss = 0
            with torch.no_grad():
                for b in validation_loader:
                    logits = self.forward(b[0])
                    probabilities = torch.sigmoid(logits)
                    loss_dev = self.loss(logits.view(-1),b[1])

                    l1_regularisation_weight_dev = sum(param.abs().sum() for name, param in self.named_parameters() if "weight" in name)
                    loss_value_regularised_dev = loss_dev + l_one_weight * l1_regularisation_weight_dev

                    total_loss += loss_value_regularised_dev.item()
                    predictions = (probabilities > 0.5).int().view(-1)
                    total_correct = (predictions == b[1]).sum().item()
                    total_samples += b[1].size(0)
            print(f'Epoch {epoch}, the dev cross entropy loss is: {total_loss}')
            print(f'Epoch {epoch}, the accuracy is: {total_correct / total_samples}')




model_l1 = BaseNetworkWithL1()

#With a l1 weight of .1 the model accuracy doesn't improve. If I change it to 0.01 it does improve but is still less accurate than
#the unregularised model
model_l1.optimize(train_dl, validation_dl, 0.01)


class BaseNetworkWithL2(nn.Module):

    def __init__(self):
        super(BaseNetworkWithL2, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(30,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,1)
        )
        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, x):
        logits = self.layers(x)
        return logits
    
    def optimize(self, training_loader, validation_loader, l_one_weight, epochs = 20, learning_rate = 0.0001):
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        for epoch in range(epochs):
            self.train()
            for batch in training_loader:
                optimizer.zero_grad()
                logits = self.forward(batch[0])
                loss_value = self.loss(logits.view(-1), batch[1])
                l2_regularisation_weight = sum(torch.pow(param, 2).sum() for name, param in self.named_parameters() if "weight" in name)
                loss_value_regularised = loss_value + l_one_weight * l2_regularisation_weight
                loss_value_regularised.backward()
                optimizer.step()

            total_correct = 0
            total_samples = 0
            total_loss = 0
            with torch.no_grad():
                for b in validation_loader:
                    logits = self.forward(b[0])
                    probabilities = torch.sigmoid(logits)
                    loss_dev = self.loss(logits.view(-1),b[1])

                    l2_regularisation_weight_dev = sum(torch.pow(param, 2).sum() for name, param in self.named_parameters() if "weight" in name)
                    loss_value_regularised_dev = loss_dev + l_one_weight * l2_regularisation_weight_dev

                    total_loss += loss_value_regularised_dev.item()
                    predictions = (probabilities > 0.5).int().view(-1)
                    total_correct = (predictions == b[1]).sum().item()
                    total_samples += b[1].size(0)
            print(f'Epoch {epoch}, the dev cross entropy loss is: {total_loss}')
            print(f'Epoch {epoch}, the accuracy is: {total_correct / total_samples}')




model_l2 = BaseNetworkWithL2()

model_l2.optimize(train_dl, validation_dl, 0.1)

from visualize_model import visualize_model

visualize_model(model_early_stop, "Early Stop")
visualize_model(model_l1, "L1")
visualize_model(model_l2, "L2")
