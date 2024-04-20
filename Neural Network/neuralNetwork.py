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
from utils import getSeasonData, createTestSetLocation, pickleResults

# from torchsummary import summary
import openpyxl

pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)


class neuralNetwork(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(neuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc1_dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc2_dropout = nn.Dropout(p=0.2)
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
        self.data = torch.tensor(
            dataFrame.drop("winnerATS", axis=1).values, dtype=torch.float
        )
        self.target = torch.tensor(
            dataFrame["winnerATS"].values, dtype=torch.float
        ).unsqueeze(1)

    # This gets one row, ground truth https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST
    def __getitem__(self, index):
        return (self.data[index], self.target[index])

    def __len__(self):
        return self.data.shape[0]


def get_data(gameStats, teamStats):

    # Set data to use
    indexStats = [
        "seasonNumber",
        "week",
        "home",
        "away",
    ]  # Stats to index database by (not used in network)

    # Choose which stats to normalize
    normalizedStats = ["TotalYards", "Turnovers", "CurrWins", "spread"]

    # Add prefix to team stats and normalize stats, set dataColumns to be columns pulled from pp_master database

    # Build dataColumns for columns used when reading database
    dataColumns = []
    dataColumns += indexStats  # must read index stats from database
    dataColumns += gameStats  # must read game stats from database
    for info in teamStats:
        for loc in ["home", "away"]:
            dataColumns.append(loc + info)

    # Build normalizedColumns for columns to be normalized in database
    normalizedColumns = []
    for info in normalizedStats:
        if info in teamStats:
            for loc in ["home", "away"]:
                normalizedColumns.append(loc + info)
        else:
            normalizedColumns.append(info)

    # Read database
    filePath = os.path.join(os.getcwd(), "pp_master_boxscore_data_1970_2019.csv")
    database = pd.read_csv(
        filePath,
        usecols=dataColumns,
        index_col=["seasonNumber", "home", "away", "week"],
    )

    # store all stats used in network, including target
    usedStats = list(database.columns)

    # Found bug, winner column not generating properly
    # database[((database['homeScore'] > database['awayScore']) & ((database['winner'] == 0))) | ((database['homeScore'] < database['awayScore']) & ((database['winner'] == 1)))]

    # Normalize stats in database
    for info in normalizedColumns:
        database[info] = (database[info] - database[info].mean()) / database[info].std()

    return database


def train_model(model, trainLoader, testLoader, loss, optimizer, epochs):
    # Set train and test loss and accuracy to be appended to after each epoch
    epochTrainLoss = []
    epochTestLoss = []
    epochTestAccuracy = []
    epochTrainAccuracy = []

    # Initialize variables to hold max accuracy of the test set
    maxAccuracy = 0
    maxAccuracyEpoch = 0

    sigmoid = nn.Sigmoid()

    for i in range(epochs):
        batchTrainLoss = []
        batchTestLoss = []
        totalTrain = 0
        correctTrain = 0
        for j, (data, target) in enumerate(trainLoader):
            model.train()
            # Zero gradient and then predict
            optimizer.zero_grad()
            pred = model(data)
            pred = sigmoid(pred)

            # Compute loss
            trainLoss = loss(pred, target)

            # Append loss to list containing all losses in batch
            batchTrainLoss.append(trainLoss.item())

            # Calculate total number of batches run and total correct
            totalTrain += target.size(0)
            correctTrain += (torch.round(pred) == target).sum().item()

            # propagate gradients and step in SGD
            trainLoss.backward()
            optimizer.step()

        # Use average batch loss to save to total training loss for this epoch
        epochTrainLoss.append(np.mean(batchTrainLoss))
        trainAccuracy = correctTrain / totalTrain
        epochTrainAccuracy.append(trainAccuracy)
        # trainError.append(torch.mean((torch.round(pred) == target).float()))

        totalTest = 0
        correctTest = 0
        with torch.no_grad():
            for k, (data, target) in enumerate(testLoader):
                model.eval()
                # Compute prediction
                pred = model(data)
                pred = sigmoid(pred)

                # Compute loss and append to list for this batch
                testLoss = loss(pred, target)
                batchTestLoss.append(testLoss.item())

                # Calculate total number of batches run and total correct
                totalTest += target.size(0)
                correctTest += (torch.round(pred) == target).sum().item()

            # testLoss.append(torch.mean((torch.round(pred) == target).float()))

        epochTestLoss.append(np.mean(batchTestLoss))
        testAccuracy = correctTest / totalTest
        epochTestAccuracy.append(testAccuracy)
        if (
            testAccuracy > maxAccuracy and i > 3
        ):  # Don't save accuracy for first few epochs (they are random)
            maxAccuracy = testAccuracy
            maxAccuracyEpoch = i

        if i % 100 == 0:
            print("Training Loss: " + str(np.mean(batchTrainLoss)))
            print("Training Accuracy: " + str(trainAccuracy))
            print("Testing Loss: " + str(np.mean(batchTestLoss)))
            print("Testing Accuracy: " + str(testAccuracy))
            print("-----------------------------------------")

    plt.plot(epochTrainLoss)
    plt.plot(epochTrainAccuracy)
    plt.plot(epochTestLoss)
    plt.plot(epochTestAccuracy)
    plt.legend(["Training Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])

    resultsDict = {
        "trainLoss": epochTrainLoss,
        "trainAccuracy": epochTrainAccuracy,
        "testLoss": epochTestLoss,
        "testAccuracy": epochTestAccuracy,
        "maxAccuracy": maxAccuracy,
        "maxAccuracyEpoch": maxAccuracyEpoch,
    }

    return resultsDict


def save_run():
    # Save model
    for i in range(1000):
        modelFolder = "./Model Info/Model #{}".format(i)
        if not os.path.exists(modelFolder):
            plotName = "Model #{}.png".format(i)

            os.makedirs(modelFolder)

            plt.savefig(os.path.join(modelFolder, plotName))
            plt.savefig("./Model Info/{}".format(plotName))
            torch.save(
                model.state_dict(),
                os.path.join(modelFolder, plotName.replace(".png", ".pt")),
            )
            pickleResults(resultsDict, modelFolder, i)
            break

    print(
        "Max Accuracy: {} at {}".format(
            resultsDict["maxAccuracy"], resultsDict["maxAccuracyEpoch"]
        )
    )

    # Allow user input from console for model run notes, save all information about run into csv file

    notes = input("Notes about model?\n")
    with open("./Model Info/models.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                plotName,
                "-".join(map(str, trainYears)),
                "-".join(map(str, testYears)),
                learning_rate,
                epochs,
                batchAmount,
                resultsDict["maxAccuracy"],
                resultsDict["maxAccuracyEpoch"],
                preGameCount,
                model,
                gameStats + teamStats,
                notes,
            ]
        )

    plt.show()


if __name__ == "__main__":
    gameStats = ["spread", "winnerATS"]  # Game data (used in network)
    teamStats = [
        "CurrWins",
        "TotalYards",
        "Turnovers",
    ]  # Stats from both teams; prefix of "home" or "away" in database (used in network)

    database = get_data(gameStats, teamStats)

    # Extract training and test sets
    # (originalTestData and originalTrainData hold raw data, testData and trainData hold calculated values done in future steps)
    trainYears = (2004, 2005)
    for year in range(2008, 2011):
        testYears = (year, year)
        originalTrainData = getSeasonData(database, trainYears)
        originalTestData = getSeasonData(database, testYears)
        trainData = originalTrainData.copy(deep=True)
        testData = originalTestData.copy(deep=True)

        # Build trainData and testData datasets using previous "preGameCount" number of games for each team
        preGameCount = 4
        print(
            "Creating Train and Test Datasets using past {} games...".format(
                preGameCount
            )
        )
        start = time.time()
        trainData = createTestSetLocation(database, trainData, preGameCount)
        testData = createTestSetLocation(database, testData, preGameCount)
        print("Datasets Complete!! Creation took: {}s".format(time.time() - start))

        # Set train and test loader to be used in training/test loops
        batchAmount = 3
        trainLoader = DataLoader(
            NFLDataset(trainData), batch_size=len(trainData) // batchAmount
        )
        testLoader = DataLoader(
            NFLDataset(testData), batch_size=len(testData) // batchAmount
        )

        # loss = nn.CrossEntropyLoss()
        # https://www.youtube.com/watch?v=wpPkDSMzdKo>
        # weight = torch.tensor(np.arange(0, 1, 1/777)).unsqueeze(1)
        # loss = nn.BCELoss(weight=weight)

        # Build model from neuralNetwork class and set training parameters
        model = neuralNetwork(7, 75, 75, 1)
        loss = nn.BCELoss()
        learning_rate = 0.05
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            nesterov=True,
            momentum=0.9,
            dampening=0,
            weight_decay=0.01,
        )
        epochs = 1000

        # Train model
        resultsDict = train_model(
            model, trainLoader, testLoader, loss, optimizer, epochs
        )

        save_run()
