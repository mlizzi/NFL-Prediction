import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import datetime as dt
from matplotlib import pyplot as plt
import time
import csv
# from torchsummary import summary
import openpyxl
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)

SEASON_TO_YEAR = dict(zip(range(1970, 2020), range(51,101)))

class neuralNetwork(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(neuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc1_dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc2_dropout = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc1_dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc2_dropout(out)
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

def getSeasonData(database, years):
    # database is indexed by season number, this function allows indexing by year instead
    # Note: Could not use calendar year directly due to some games in a season bleeding over into the next calendar year
    startSeasonYear = SEASON_TO_YEAR[years[0]]
    endSeasonYear = SEASON_TO_YEAR[years[1]]

    return database.loc[startSeasonYear: endSeasonYear]

def createTestSet(testData, preGameCount):
    for idx, row in testData.iterrows():
        indexInDatabase = int(list(database.index).index(idx))

        filteredHomeDatabase = database.reset_index()[(database.reset_index().home == idx[1])]# | (database.reset_index().away == idx[1])]
        homeValue = filteredHomeDatabase.index.get_loc(indexInDatabase)
        pastHomeGames = filteredHomeDatabase.iloc[homeValue - preGameCount:homeValue]

        filteredAwayDatabase = database.reset_index()[(database.reset_index().away == idx[2])]# | (database.reset_index().home == idx[2])]
        awayValue = filteredAwayDatabase.index.get_loc(indexInDatabase)
        pastAwayGames = filteredAwayDatabase.iloc[awayValue - preGameCount:awayValue]

        # b = database[(database.home=='New England Patriots') | (database.away == 'New England Patriots')]
        # pd.concat([database.loc[90,'New England Patriots',:],database.loc[90,:,'New England Patriots']]).reset_index()

        for stat, val in row.iteritems():
            if 'home' in stat:
                testData.at[idx, stat] = pastHomeGames[stat].mean()
            elif 'away' in stat:
                testData.at[idx, stat] = pastAwayGames[stat].mean()
    return testData

if __name__ == '__main__':

    # Set data to use, which data to normalize (teamStats contains team specific data and prefix with "home" or "away"
    gameStats = ['week', 'home', 'away', 'seasonNumber', 'winnerATS']
    teamStats = ['PassYds', 'RushYds', 'PassTDs', 'Sacks', 'Turnovers']
    normalizedStats = teamStats

    # Add prefix to team stats and normalize stats, set dataColumns to be columns pulled from pp_master database
    normalizedColumns = []
    dataColumns = []
    for info in teamStats:
        for loc in ['home', 'away']:
            dataColumns.append(loc + info)

    for info in normalizedStats:
        for loc in ['home', 'away']:
            normalizedColumns.append(loc + info)

    dataColumns += gameStats # dataColumns holds column names to pull from database

    # Read database
    filePath = os.path.join(os.getcwd(), 'pp_master_boxscore_data_1970_2019.csv')
    database = pd.read_csv(filePath, usecols=dataColumns, index_col=['seasonNumber', 'home', 'away', 'week'])

    # Normalize stats
    for info in normalizedColumns:
        database[info] = (database[info] - database[info].mean()) / database[info].std()

    # Extract training and test sets
    # (originalTestData holds original data, testData to be adjusted to hold calculated data in future steps)
    trainYears = (2003, 2003)
    testYears = (2003, 2003)
    trainData = getSeasonData(database, trainYears)[:213]
    originalTestData = getSeasonData(database, testYears)[213:]

    testData = originalTestData.copy(deep=True)


    # Build testing dataset using previous preGameCount number of games for each team
    preGameCount = 4
    print("Creating Test Dataset using past {} games...".format(preGameCount))
    start = time.time()
    # testData = createTestSet(testData, preGameCount)
    print("Test Dataset Complete!! Creation took: {}s".format(time.time() - start))

    # Set train and test loader to be used in training/test loops
    trainLoader = DataLoader(NFLDataset(trainData), batch_size=len(trainData)//2)
    testLoader = DataLoader(NFLDataset(testData), batch_size=len(testData)//2)

    # Build model from neuralNetwork class
    model = neuralNetwork(10, 200, 50, 1)


    # loss = nn.CrossEntropyLoss()
    # https://www.youtube.com/watch?v=wpPkDSMzdKo>
    # weight = torch.tensor(np.arange(0, 1, 1/777)).unsqueeze(1)
    # loss = nn.BCELoss(weight=weight)
    loss = nn.BCELoss()
    learning_rate = 0.005
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0)
    x = nn.Sigmoid()

    # Set train and test loss and accuracy to be appended to after each epoch
    trainAccuracy = []
    epochTrainLoss = []
    testAccuracy = []
    epochTestLoss = []

    # Initialize variables to hold max accuracy of the test set
    maxAccuracy = 0
    maxAccuracyEpoch = 0

    epochs = 4000
    for i in range(epochs):
        batchTrainLoss = []
        batchTestLoss = []
        for j, (data, target) in enumerate(trainLoader):
            # Zero gradient and then predict
            optimizer.zero_grad()
            pred = model(data)
            pred = x(pred)

            # Compute loss
            trainLoss = loss(pred, target)

            # Append loss to list containing all losses in batch
            batchTrainLoss.append(trainLoss.item())

            # propagate gradients and step in SGD
            trainLoss.backward()
            optimizer.step()

        # Use average batch loss to save to total training loss for this epoch
        epochTrainLoss.append(np.mean(batchTrainLoss))
        # trainError.append(torch.mean((torch.round(pred) == target).float()))


        total = 0
        correct = 0
        with torch.no_grad():
            for k, (data, target) in enumerate(testLoader):

                # Compute prediction
                pred = model(data)
                pred = x(pred)

                # Compute loss and append to list for this batch
                testLoss = loss(pred, target)
                batchTestLoss.append(testLoss.item())

                # Calculate total number of batches run and total correct
                total += target.size(0)
                correct += (torch.round(pred) == target).sum().item()

            # testLoss.append(torch.mean((torch.round(pred) == target).float()))

        epochTestLoss.append(np.mean(batchTestLoss))
        testAccuracy = correct / total
        if testAccuracy > maxAccuracy:
            maxAccuracy = testAccuracy
            maxAccuracyEpoch = i

        if i % 100 == 0:
            print("Training Loss: " + str(trainLoss.item()))
            print("Testing Loss: " + str(testLoss.item()))
            print("Testing Accuracy: " + str(testAccuracy))
            print('-----------------------------------------')


    plt.plot(epochTestLoss)
    plt.plot(epochTrainLoss)
    plt.legend(['Test Loss', 'Training Loss'])

    plotPath = os.path.join(os.getcwd(), 'Model Plots')
    for i in range(1000):
        plotName = 'Model #{}.png'.format(i)
        if plotName not in os.listdir(plotPath):
            plt.savefig(os.path.join(plotPath, plotName))
            torch.save(model.state_dict(),os.path.join(plotPath, plotName.replace('.png', '.pt')))
            break

    print('Max Accuracy: {} at {}'.format(maxAccuracy, maxAccuracyEpoch))
    notes = input("Notes about model?\n")

    # workbook = xlsxwriter.Workbook()
    # book = openpyxl.load_workbook('./Model Plots/model.xlsx')
    # worksheet = workbook.add_worksheet()
    with open('./Model Plots/models.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([plotName, "-".join(map(str, trainYears)), "-".join(map(str, testYears)), learning_rate, epochs, maxAccuracy, maxAccuracyEpoch, preGameCount, notes])

    plt.show()

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

