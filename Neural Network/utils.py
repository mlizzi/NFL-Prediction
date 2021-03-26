import os
import pandas as pd
import pickle
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

SEASON_TO_YEAR = dict(zip(range(1970, 2020), range(51, 101)))

def getSeasonData(database, years):
    # database is indexed by season number, this function allows indexing by year instead
    # Note: Could not use calendar year directly due to some games in a season bleeding over into the next calendar year
    startSeasonYear = SEASON_TO_YEAR[years[0]]
    endSeasonYear = SEASON_TO_YEAR[years[1]]

    return database.loc[startSeasonYear: endSeasonYear]

def createTestSetLocation(database, testData, preGameCount):
    indexToDelete = []
    for idx, row in testData.iterrows():

        indexInDatabase = int(list(database.index).index(idx))

        filteredHomeDatabase = database.reset_index()[(database.reset_index().home == idx[1])]# | (database.reset_index().away == idx[1])]
        homeValue = filteredHomeDatabase.index.get_loc(indexInDatabase)
        pastHomeGames = filteredHomeDatabase.iloc[homeValue - preGameCount:homeValue]

        filteredAwayDatabase = database.reset_index()[(database.reset_index().away == idx[2])]# | (database.reset_index().home == idx[2])]
        awayValue = filteredAwayDatabase.index.get_loc(indexInDatabase)
        pastAwayGames = filteredAwayDatabase.iloc[awayValue - preGameCount:awayValue]

        if len(pastHomeGames) != preGameCount or len(pastAwayGames) != preGameCount:
            print("Error: No previous games found for game {}. Game is being deleted...".format(idx))
            indexToDelete.append(idx)
            continue
            # b = database[(database.home=='New England Patriots') | (database.away == 'New England Patriots')]
            # pd.concat([database.loc[90,'New England Patriots',:],database.loc[90,:,'New England Patriots']]).reset_index()

        for stat, val in row.iteritems():
            if 'CurrWin' in stat or 'spread' in stat:
                continue

            if 'home' in stat:
                testData.at[idx, stat] = pastHomeGames[stat].mean()
            elif 'away' in stat:
                testData.at[idx, stat] = pastAwayGames[stat].mean()
    testData = testData.drop(indexToDelete)
    return testData

def createTestSetPrevious(database, testData, preGameCount):
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

def pickleTestSet(preGameCount, testYears):
    ######################## SETUP STAGE ########################
    # Set data to use
    indexStats = ['seasonNumber', 'home', 'away', 'week']  # Stats to index database by (not used in network)
    gameStats = ['spread', 'winnerATS']  # Game data (used in network)
    teamStats = ['CurrWins', 'TotalYards', 'Turnovers']  # Stats from both teams; prefix of "home" or "away" in database (used in network)

    # Choose which stats to normalize
    normalizedStats = ['TotalYards', 'Turnovers', 'CurrWins', 'spread']

    # Add prefix to team stats and normalize stats, set dataColumns to be columns pulled from pp_master database

    # Build dataColumns for columns used when reading database
    dataColumns = []
    dataColumns += indexStats  # must read index stats from database
    dataColumns += gameStats  # must read game stats from database
    for info in teamStats:
        for loc in ['home', 'away']:
            dataColumns.append(loc + info)

    # Build normalizedColumns for columns to be normalized in database
    normalizedColumns = []
    for info in normalizedStats:
        if info in teamStats:
            for loc in ['home', 'away']:
                normalizedColumns.append(loc + info)
        else:
            normalizedColumns.append(info)

    # Read database
    filePath = os.path.join(os.getcwd(), 'pp_master_boxscore_data_1970_2019.csv')
    database = pd.read_csv(filePath, usecols=dataColumns, index_col=['seasonNumber', 'home', 'away', 'week'])

    # Found bug, winner column not generating properly
    # database[((database['homeScore'] > database['awayScore']) & ((database['winner'] == 0))) | ((database['homeScore'] < database['awayScore']) & ((database['winner'] == 1)))]

    # Normalize stats in database
    for info in normalizedColumns:
        database[info] = (database[info] - database[info].mean()) / database[info].std()

    for year in range(testYears[0], testYears[1]+1):
        print("Pickling year {}, pregame count of {} ... ".format(year, preGameCount))
        originalTestData = getSeasonData(database, (year, year))
        testData = originalTestData.copy(deep=True)
        testData = createTestSetLocation(database, testData, preGameCount)
        if not os.path.exists("Pickled_Test_Data"):
            os.makedirs("Pickled_Test_Data")
        outfile = open("./Pickled_Test_Data/{0}_test_data_pgc_{1}.pkl".format(year, preGameCount), 'wb')
        pickle.dump(testData, outfile)
        outfile.close()

def pickleResults(resultsDict, modelFolder, modelNumber):

    for key, value in resultsDict.items():
        outfile = open("./{0}/{1}{2}.pkl".format(modelFolder, key, modelNumber), 'wb')
        pickle.dump(value, outfile)
        outfile.close()


def loadPickledFiles(years, preGameCount):
    database = pd.DataFrame()

    for year in range(years[0],years[1]+1):
        for file in os.listdir('./Pickled_Test_Data'):
            if file == "{}_test_data_pgc_{}.pkl".format(year, preGameCount):
                data = pickle.load(open('./Pickled_Test_Data/{}'.format(file), 'rb'))
                database = database.append(data)
                break

    return database

def oldPlotMethod():
    pass
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

if __name__ == '__main__':
    print('finding')
    # pickleTestSet(3, (1971, 2000))
    # database = loadPickledFiles((2006, 2019), 3)
    print('done')
    # print(database)
    pass

