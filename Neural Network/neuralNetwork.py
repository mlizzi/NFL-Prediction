import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import datetime as dt
from matplotlib import pyplot as plt

class neuralNetwork(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(neuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

class NFLDataset(Dataset):

    def __init__(self, data):
        self.data = torch.from_numpy(data.to_numpy(dtype=np.float32))

    # This gets one row, ground truth https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST
    def __getitem__(self, index):
        return (self.data[index][1:], self.data[index][0].unsqueeze(0))

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':

    data = ['RushYds', 'PassTDs', 'Sacks', 'Fumbles']

    filePath = os.path.join(os.getcwd(), 'pp_master_boxscore_data_1970_2019.csv')


    database = pd.read_csv(filePath, parse_dates=['Date'], index_col='seasonNumber', usecols=)

    tempDatabase = database[database.columns.difference(['Home Line Close', 'Home Score', 'Away Score', 'Home Yards nonnorm', 'Away Yards nonnorm', 'Home Team', 'Away Team', 'Playoff Game?'])]
    database = tempDatabase

    trainStartYear = 2006
    trainEndYear = 2007
    trainData = database.loc[str(trainStartYear): str(trainEndYear)]
    testStartYear = 2008
    testEndYear = 2009
    testData = database.loc[str(testStartYear):str(testEndYear)]
    # testData = database.loc[database.index.isin(testGames)]

    trainLoader = DataLoader(NFLDataset(trainData), batch_size=len(trainData))
    testLoader = DataLoader(NFLDataset(testData), batch_size=len(testData))

    fig, axs = plt.subplots(2,1, figsize=(10,10))
    for t in range(5): # Testing
        model = neuralNetwork(5, 3, 2, 1)

        # loss = nn.CrossEntropyLoss()
        # https://www.youtube.com/watch?v=wpPkDSMzdKo>
        loss = nn.BCELoss()
        learning_rate = 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0)
        x = nn.Sigmoid()
        error = []
        testAccuracy = []
        for i in range(1500):
            for j, (data, target) in enumerate(trainLoader):
                optimizer.zero_grad()
                pred = model(data)
                pred = x(pred)

                output = loss(pred, target)
                #output.item to save loss

                output.backward()
                optimizer.step()

            error.append(torch.mean((torch.round(pred) == target).float()))



        total = 0
        correct = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(testLoader):

                pred = model(data)
                pred = x(pred)
                output = loss(pred, target)
                total += target.size(0)
                pred = torch.round(pred)
                correct += (pred == target).sum().item()

        accuracy = correct / total


        axs[0].plot(error, label='Model #' + str(t+1))
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Training Accuracy')

        axs[1].scatter(t+1, accuracy, label='Model #' + str(t+1))
        axs[1].set_xlabel('Model #')
        axs[1].set_ylabel('Test Accuracy')


        # testAccuracy.append(accuracy)
        print('Accuracy of the network on test set: Games from {0}-{1}: {2}%'.format(testStartYear, testEndYear, accuracy))


    axs[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1))

    axs[0].title.set_text('Training accuracy for different models (Games from {0}-{1})'.format(trainStartYear, trainEndYear))
    axs[1].title.set_text('Test accuracy for different models (Games from {0}-{1})'.format(testStartYear, testEndYear))

    axs[1].set_xticks(range(1, t+2))

    plt.tight_layout()
    plt.savefig('Multiple Training Experiment')

