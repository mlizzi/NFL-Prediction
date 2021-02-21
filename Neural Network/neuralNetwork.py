import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import datetime as dt
from matplotlib import pyplot as plt
# from torchsummary import summary

SEASON_TO_YEAR = dict(zip(range(1970, 2020), range(51,101)))

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

    def __init__(self, dataFrame):
        self.data = torch.tensor(dataFrame.drop('winnerATS', axis=1).values, dtype=torch.float)
        self.target = torch.tensor(dataFrame['winnerATS'].values, dtype=torch.float).unsqueeze(1)
    # This gets one row, ground truth https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST
    def __getitem__(self, index):
        return (self.data[index], self.target[index])

    def __len__(self):
        return self.data.shape[0]

def getSeasonData(database, startSeason, endSeason):
    startSeasonYear = SEASON_TO_YEAR[startSeason]
    endSeasonYear = SEASON_TO_YEAR[endSeason]

    return database.loc[startSeasonYear: endSeasonYear]
if __name__ == '__main__':


    gameStats = ['seasonNumber', 'winnerATS']
    teamStats = ['PassYds', 'RushYds', 'PassTDs', 'Sacks', 'Turnovers']

    normalizedStats = teamStats

    normalizedColumns = []
    dataColumns = []
    for info in teamStats:
        for loc in ['home', 'away']:
            dataColumns.append(loc + info)

    for info in normalizedStats:
        for loc in ['home', 'away']:
            normalizedColumns.append(loc + info)



    dataColumns += gameStats

    filePath = os.path.join(os.getcwd(), 'pp_master_boxscore_data_1970_2019.csv')


    # database = pd.read_csv(filePath, index_col=['seasonNumber'], usecols=dataColumns)
    database = pd.read_csv(filePath, usecols=dataColumns, index_col='seasonNumber')

    for info in normalizedColumns:
        database[info] = (database[info] - database[info].mean()) / database[info].std()

    trainData = getSeasonData(database, 2002, 2003)
    testData = getSeasonData(database, 2004, 2004)

    trainLoader = DataLoader(NFLDataset(trainData), batch_size=len(trainData))
    testLoader = DataLoader(NFLDataset(testData), batch_size=len(testData))

    model = neuralNetwork(10, 5, 2, 1)

    # loss = nn.CrossEntropyLoss()
    # https://www.youtube.com/watch?v=wpPkDSMzdKo>

    # weight = torch.tensor(np.arange(0, 1, 1/777)).unsqueeze(1)
    # loss = nn.BCELoss(weight=weight)
    loss = nn.BCELoss()
    learning_rate = 0.012
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0)
    x = nn.Sigmoid()
    error = []
    testAccuracy = []
    outputLoss = []
    for i in range(4000):
        for j, (data, target) in enumerate(trainLoader):

            optimizer.zero_grad()
            pred = model(data)
            pred = x(pred)
            output = loss(pred, target)
            #output.item to save loss
            outputLoss.append(output)
            output.backward()
            optimizer.step()

        error.append(torch.mean((torch.round(pred) == target).float()))

        if i % 100 == 0:
            print("Training Loss: " + str(output.item()))

    plt.plot(error)
    plt.plot(outputLoss)
    plt.legend(['Training Accuracy', 'Training Loss'])
    plt.savefig('Test 8')
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
#
    print(accuracy)
#     axs[0].plot(error, label='Model #' + str(t+1))
#     axs[0].set_xlabel('Epochs')
#     axs[0].set_ylabel('Training Accuracy')
#
#     axs[1].scatter(t+1, accuracy, label='Model #' + str(t+1))
#     axs[1].set_xlabel('Model #')
#     axs[1].set_ylabel('Test Accuracy')
#
#
#     # testAccuracy.append(accuracy)
#     print('Accuracy of the network on test set: Games from {0}-{1}: {2}%'.format(testStartYear, testEndYear, accuracy))
#
#
# axs[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
# axs[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
#
# axs[0].title.set_text('Training accuracy for different models (Games from {0}-{1})'.format(trainStartYear, trainEndYear))
# axs[1].title.set_text('Test accuracy for different models (Games from {0}-{1})'.format(testStartYear, testEndYear))
#
# axs[1].set_xticks(range(1, t+2))

# plt.tight_layout()
# plt.savefig('Multiple Taining Experiment')

