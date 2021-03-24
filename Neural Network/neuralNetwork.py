import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import pickle
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
pd.set_option('display.max_rows', 20)


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

def createTestSetLocation(testData, preGameCount):
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
            if 'CurrWin' in stat or 'spread' in stat:
                continue

            if 'home' in stat:
                testData.at[idx, stat] = pastHomeGames[stat].mean()
            elif 'away' in stat:
                testData.at[idx, stat] = pastAwayGames[stat].mean()
    return testData

def createTestSetPrevious(testData, preGameCount):
    for idx, row in testData.iterrows():


        indexInDatabase = int(list(database.index).index(idx))

        filteredHomeDatabase = database.reset_index()[(database.reset_index().home == idx[1]) | (database.reset_index().away == idx[1])]
        homeValue = filteredHomeDatabase.index.get_loc(indexInDatabase)
        pastHomeGames = filteredHomeDatabase.iloc[homeValue - preGameCount:homeValue]

        filteredAwayDatabase = database.reset_index()[(database.reset_index().away == idx[2]) | (database.reset_index().home == idx[2])]
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

#     ######################## SETUP STAGE ########################
#     # Set data to use
#     indexStats = ['week', 'home', 'away', 'seasonNumber'] # Stats to index database by (not used in network)
#     gameStats = ['spread', 'winnerATS'] # Game data (used in network)
#     teamStats = ['CurrWins', 'TotalYards', 'Turnovers'] # Stats from both teams; prefix of "home" or "away" in database (used in network)
#
#     # Choose which stats to normalize
#     normalizedStats = ['TotalYards', 'Turnovers', 'CurrWins', 'spread']
#
#     # Add prefix to team stats and normalize stats, set dataColumns to be columns pulled from pp_master database
#
#     # Build dataColumns for columns used when reading database
#     dataColumns = []
#     dataColumns += indexStats # must read index stats from database
#     dataColumns += gameStats  # must read game stats from database
#     for info in teamStats:
#         for loc in ['home', 'away']:
#             dataColumns.append(loc + info)
#
#     # Build normalizedColumns for columns to be normalized in database
#     normalizedColumns = []
#     for info in normalizedStats:
#         if info in teamStats:
#             for loc in ['home', 'away']:
#                 normalizedColumns.append(loc + info)
#         else:
#             normalizedColumns.append(info)
#
#     # Read database
#     filePath = os.path.join(os.getcwd(), 'pp_master_boxscore_data_1970_2019.csv')
#     database = pd.read_csv(filePath, usecols=dataColumns, index_col=['seasonNumber', 'home', 'away', 'week'])
#
#     # Found bug, winner column not generating properly
#     # database[((database['homeScore'] > database['awayScore']) & ((database['winner'] == 0))) | ((database['homeScore'] < database['awayScore']) & ((database['winner'] == 1)))]
#
#     # Normalize stats in database
#     for info in normalizedColumns:
#         database[info] = (database[info] - database[info].mean()) / database[info].std()
#
#     # Extract training and test sets
#     # (originalTestData and originalTrainData hold raw data, testData and trainData hold calculated values done in future steps)
#     trainYears = (2004, 2005)
#     for year in range(2006, 2011):
#         testYears = (year,year)
#         originalTrainData = getSeasonData(database, trainYears)
#         originalTestData = getSeasonData(database, testYears)
#         trainData = originalTrainData.copy(deep=True)
#         testData = originalTestData.copy(deep=True)
#
#         # Build trainData and testData datasets using previous "preGameCount" number of games for each team
#         preGameCount = 2
#         print("Creating Train and Test Datasets using past {} games...".format(preGameCount))
#         start = time.time()
#         trainData = createTestSetLocation(trainData, preGameCount)
#         testData = createTestSetLocation(testData, preGameCount)
#         print("Datasets Complete!! Creation took: {}s".format(time.time() - start))
#
#         # Set train and test loader to be used in training/test loops
#         batchAmount = 2
#         trainLoader = DataLoader(NFLDataset(trainData), batch_size=len(trainData)//batchAmount)
#         testLoader = DataLoader(NFLDataset(testData), batch_size=len(testData)//batchAmount)
#
#         # Build model from neuralNetwork class
#         model = neuralNetwork(7, 75, 75, 1)
#
#         # loss = nn.CrossEntropyLoss()
#         # https://www.youtube.com/watch?v=wpPkDSMzdKo>
#         # weight = torch.tensor(np.arange(0, 1, 1/777)).unsqueeze(1)
#         # loss = nn.BCELoss(weight=weight)
#
#         # Set loss, optimizer, and output activation
#         loss = nn.BCELoss()
#         learning_rate = 0.005
#         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0, weight_decay=0.01)
#         x = nn.Sigmoid()
#
#         # Set train and test loss and accuracy to be appended to after each epoch
#         epochTrainLoss = []
#         epochTestLoss = []
#         epochTestAccuracy = []
#         epochTrainAccuracy = []
#
#         # Initialize variables to hold max accuracy of the test set
#         maxAccuracy = 0
#         maxAccuracyEpoch = 0
#
#         epochs = 2000
#         for i in range(epochs):
#             batchTrainLoss = []
#             batchTestLoss = []
#             for j, (data, target) in enumerate(trainLoader):
#                 model.train()
#                 # Zero gradient and then predict
#                 optimizer.zero_grad()
#                 pred = model(data)
#                 pred = x(pred)
#
#                 # Compute loss
#                 trainLoss = loss(pred, target)
#
#                 # Append loss to list containing all losses in batch
#                 batchTrainLoss.append(trainLoss.item())
#
#                 # propagate gradients and step in SGD
#                 trainLoss.backward()
#                 optimizer.step()
#
#             # Use average batch loss to save to total training loss for this epoch
#             epochTrainLoss.append(np.mean(batchTrainLoss))
#             # trainError.append(torch.mean((torch.round(pred) == target).float()))
#
#
#             total = 0
#             correct = 0
#             with torch.no_grad():
#                 for k, (data, target) in enumerate(testLoader):
#                     model.eval()
#                     # Compute prediction
#                     pred = model(data)
#                     pred = x(pred)
#
#                     # Compute loss and append to list for this batch
#                     testLoss = loss(pred, target)
#                     batchTestLoss.append(testLoss.item())
#
#                     # Calculate total number of batches run and total correct
#                     total += target.size(0)
#                     correct += (torch.round(pred) == target).sum().item()
#
#                 # testLoss.append(torch.mean((torch.round(pred) == target).float()))
#
#             epochTestLoss.append(np.mean(batchTestLoss))
#             testAccuracy = correct / total
#             epochTestAccuracy.append(testAccuracy)
#             if testAccuracy > maxAccuracy and i > 3: # Don't save accuracy for first few epochs (they are random)
#                 maxAccuracy = testAccuracy
#                 maxAccuracyEpoch = i
#
#             if i % 100 == 0:
#                 print("Training Loss: " + str(np.mean(batchTrainLoss)))
#                 print("Testing Loss: " + str(np.mean(batchTestLoss)))
#                 print("Testing Accuracy: " + str(testAccuracy))
#                 print('-----------------------------------------')
#
#         plt.plot(epochTrainLoss)
#         plt.plot(epochTestLoss)
#         plt.plot(epochTestAccuracy)
#         plt.legend(['Training Loss', 'Test Loss', 'Test Accuracy'])
#
#
#         # save plots to path
#         plotPath = os.path.join(os.getcwd(), 'Model Plots')
#         for i in range(1000):
#             plotName = 'Model #{}.png'.format(i)
#             if plotName not in os.listdir(plotPath):
#                 plt.savefig(os.path.join(plotPath, plotName))
#                 torch.save(model.state_dict(),os.path.join(plotPath, plotName.replace('.png', '.pt')))
#                 break
#
#         print('Max Accuracy: {} at {}'.format(maxAccuracy, maxAccuracyEpoch))
#
#
#         # Allow user input from console for model run notes, save all information about run into csv file
#         notes = input("Notes about model?\n")
#         with open('./Model Plots/models.csv', 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([plotName, "-".join(map(str, trainYears)), "-".join(map(str, testYears)), learning_rate, epochs, batchAmount, maxAccuracy, maxAccuracyEpoch, preGameCount, model, gameStats + teamStats, notes])
#
#         plt.show()
#
# #     axs[0].plot(error, label='Model #' + str(t+1))
# #     axs[0].set_xlabel('Epochs')
# #     axs[0].set_ylabel('Training Accuracy')
# #
# #     axs[1].scatter(t+1, accuracy, label='Model #' + str(t+1))
# #     axs[1].set_xlabel('Model #')
# #     axs[1].set_ylabel('Test Accuracy')
# #
# #
# #     # testAccuracy.append(accuracy)
# #     print('Accuracy of the network on test set: Games from {0}-{1}: {2}%'.format(testStartYear, testEndYear, accuracy))
# #
# #
# # axs[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
# # axs[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
# #
# # axs[0].title.set_text('Training accuracy for different models (Games from {0}-{1})'.format(trainStartYear, trainEndYear))
# # axs[1].title.set_text('Test accuracy for different models (Games from {0}-{1})'.format(testStartYear, testEndYear))
# #
# # axs[1].set_xticks(range(1, t+2))
#
# # plt.tight_layout()
# # plt.savefig('Multiple Taining Experiment')

