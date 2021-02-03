import requests
import bs4
import os
import re
import pandas as pd

# NOTE: 1. This script assumes pro-football-reference.com page formats are as of January 2021. Changes to the website
#          may cause script issues
#       2. For understanding the code, it may be useful to open the associated pro-football-reference (PFR hereafter)
#          page while reading through the code and comments

def extractBoxscoreLinks(seasonStartYear):
	'''
	The script reads the years/{seasonStartYear}/games.htm page on PFR and iterates over the Week-by-Week table
	to extract the boxscore links.
	 seasonStartYear: year in which the season started. The year for a game may be different than this if a game
	 				  (usually playoffs) is played in the following year. This is accounted for and noted in the code.


	 :return: weekByWeek = {link1: {statName1: statVal1, ...} }
				 link1: suffix of link to PFR page (can access by appending to http://www.pro-football-reference.com/)
				 statName#: stat extracted from Week-by-Week table
				 statVal#: value of stat
	'''

	# Read games page for specified year from pro-football-reference with soup
	result = requests.get('http://www.pro-football-reference.com/years/{0}/games.htm'.format(seasonStartYear))
	content = re.sub(r'(?m)^\<!--.*\n?', '', result.content.decode('ISO-8859-1'))
	content = re.sub(r'(?m)^\-->.*\n?', '', content)
	soup = bs4.BeautifulSoup(content, 'html5lib')

	# Find table containing all games, extract rows from tbody of table
	gamesTable = soup.find('table', id='games')
	games = gamesTable.find('tbody')
	rows = games.find_all('tr')

	# Initialize dictionary which will contain info from weekByWeek table from webpage defined in result variable
	weekByWeek = {}

	# Save season number (NFL started in 1920, considered season 1)
	seasonNumber = seasonStartYear - 1920 + 1

	# Iterate over each row in table
	for row in rows:
		# Extract hyperlink to game data.
		gameLink = row.find(attrs={'data-stat': 'boxscore_word'}).find('a')

		# If no hyperlink exists, skip this row (most likely a header row). Else save hyperlink address
		if gameLink == None:
			continue
		else:
			gameLink = gameLink['href']

		# Extract year, week, date, winner, and loser from row
		year = gameLink.split('/')[2][:4] # Extracts from gameLink to ensure proper year is used
		week = row.find(attrs={'data-stat': 'week_num'}).text
		date = row.find(attrs={'data-stat': 'game_date'}).text
		winner = row.find(attrs={'data-stat': 'winner'}).text
		loser = row.find(attrs={'data-stat': 'loser'}).text

		# Check if home/away is winner/loser based on @ symbol in table
		if row.find(attrs={'data-stat': 'game_location'}).text == '@':
			away = winner
			home = loser
		else:
			home = winner
			away = loser

		# Append date, week, home and away to weekByWeek dictionary with key of gameLink
		weekByWeek[gameLink] = {'date': date, 'week': week, 'year': year, 'seasonNumber': seasonNumber, 'home': home, 'away': away}
	return weekByWeek

def extractAllBoxscores(seasonStartYear):

	'''
	First, we extract all boxscore links from the PFR years/{yearNum}/games.htm page, storing in the dictionary
		called weekByWeek.
	Then we access each boxscore link and extract data. Currently the following stats are used for both home and away:
		1. Coach Name
		2. Current Record
		3. Roof
		4. Surface
		5. Vegas Line
		6. Over/Under
		7. Starting Rosters
	'''
	weekByWeek = extractBoxscoreLinks(seasonStartYear)

	# Initialize DataFrame where each row will be one game and each column contains different statistic
	dataDF = pd.DataFrame()

	# Iterate over all links, accessing boxscore webpage and extracting necessary data
	for gameLink, info in weekByWeek.items():
		# Initialize empty data dict which is used to store data from current gameLink on each iteration
		data = {}

		# Update data with info from weekByWeek dict
		data.update(info)

		# Read boxscore page for specified gameLink from pro-football-reference with soup
		link = 'http://www.pro-football-reference.com/{0}'.format(gameLink)
		result = requests.get('http://www.pro-football-reference.com/{0}'.format(gameLink))
		content = re.sub(r'(?m)^\<!--.*\n?', '', result.content.decode('ISO-8859-1'))
		content = re.sub(r'(?m)^\-->.*\n?', '', content)
		soup = bs4.BeautifulSoup(content, 'lxml')

		######################## Try to extract both home and away coach names ########################
		try:
			coaches = soup.find_all('div', {'class': 'datapoint'}) #'datapoint' is class name given only to coach text
			data['homeCoach'] = coaches[0].get_text().replace('Coach: ', '')
			data['awayCoach'] = coaches[1].get_text().replace('Coach: ', '')
		except:
			#TODO Add Logger
			print('Unable to find coaches for game: ' + gameLink)

		######################## Try to extract both home and away records ########################
		try:
			scores = soup.find_all('div', {'class': 'score'}) #'score' is class name given only to record text
			data['homeRecord'] = scores[0].findNext().get_text()
			data['awayRecord'] = scores[1].findNext().get_text()
		except:
			#TODO Add Logger
			print('Unable to find scores for game: ' + gameLink)

		######################## Try and extract categories from Game Info table #######################
		# Extract the following categories
		categories = ['roof', 'surface', 'vegas line', 'over/under']
		try:
			# Find Game Info table
			infoTable = soup.find('table', {'id': 'game_info'})

			# Iterate over each row in table, skip the first one since it is the table title
			for row in infoTable.find_all('tr')[1:]:
				# if 'game info' in row.get_text().lower():
				# 	continue

				# Get category name for row (found in the th tag of the table)
				category = row.find('th').get_text().lower()

				# Capture only category found in categories list
				if category in categories:
					# Get category data for row (found in the td tag of the table)
					data[category] = row.find('td').get_text()
		except:
			# TODO Add Logger
			print('Unable to find Game Info for game: ' + gameLink)

		##################### Try and extract starters from starters tables (one home, one away) ####################
		try:
			# PFR has starting roster data for 1999 -> present. For years previous, starting QB are assumed
			# to be player with most pass attempts in the game
			if seasonStartYear >= 1999:

				# Find rows in home starters table (each row is a tr tag nested in tbody tag)
				starterTable = soup.find('table', id='home_starters')
				starterTable = starterTable.find('tbody')
				rows = starterTable.find_all('tr')

				# Iterate over each row, saving playerName and position to data dictionary
				for row in rows:
					playerName = row.find('th').get_text().rstrip()
					position = row.find('td').get_text().rstrip()
					data['home' + position] = playerName

				# Find rows in away starters table (each row is a tr tag nested in tbody tag)
				starterTable = soup.find('table', id='vis_starters')
				starterTable = starterTable.find('tbody')
				rows = starterTable.find_all('tr')

				# Iterate over each row, saving playerName and position to data dictionary
				for row in rows:
					playerName = row.find('th').get_text().rstrip()
					position = row.find('td').get_text().rstrip()
					data['away' + position] = playerName

			else:
				statsTable = soup.find('table', id='player_offense')
				statsTable = statsTable.find('tbody')
				rows = statsTable.find_all('tr')

				awayPlayers = True
				QBNames = {'homeQB': '', 'awayQB': ''}
				maxAttempts = 0
				for row in rows:
					try:
						passAttempts = int(row.find('td', {'data-stat': 'pass_att'}).get_text())
					except:
						awayPlayers = False
						maxAttempts = 0
						continue

					if passAttempts == 0:
						continue
					else:
						if passAttempts > maxAttempts:
							maxAttempts = passAttempts
							playerName = row.find('th', {'data-stat': 'player'}).get_text().rstrip()
							if awayPlayers:
								data['awayQB'] = playerName
							else:
								data['homeQB'] = playerName

		except:
			print('Unable to find starting roster info for game: ' + gameLink)
			# TODO Add Logger
		######################## Try and extract team stats from Team Stats table #######################
		try:
			# Find rows in team stats starters table (each row is a tr tag nested in tbody tag)
			statsTable = soup.find('table', id='team_stats')
			statsTable = statsTable.find('tbody')
			rows = statsTable.find_all('tr')

			# Iterate over each row, saving playerName and position to data dictionary
			for row in rows:
				stat = row.find('th').get_text().rstrip()
				awayVal = row.find('td', {'data-stat': 'vis_stat'}).get_text().rstrip()
				homeVal = row.find('td', {'data-stat': 'home_stat'}).get_text().rstrip()
				data['away ' + stat] = awayVal
				data['home ' + stat] = homeVal

		except:
			print('Unable to find Team stats for game: ' + gameLink)

		# print to keep track of progress
		print('Done scrapping Week {3} Year {2}:  {0} at {1}'.format(data['home'], data['away'], data['year'], data['week']))

		# Append data from data dictionary on each iteration (aka for each game) to dataDF
		if dataDF.empty:
			dataDF = pd.DataFrame.from_dict([data])
		else:
			dataDF = dataDF.append(data, ignore_index=True)


	return dataDF

def concatBoxscoreCSVs(filePath):
	'''Concatentates all CSVs in filePath (excluding any master CSVs) '''
	# Selects all csv files in filePath that are not master
	csvFiles = [file for file in filePath if file.endswith('.csv') and 'master' not in file]
	# Concatenate all csvFiles into masterFile
	masterFile = pd.concat([pd.read_csv(f, index_col=0) for f in csvFiles])
	# Reset index (since concatenation causes indices not to increment by 1 each row)
	masterFile.reset_index(inplace=True, drop=True)
	return masterFile

def postProcessColumns(masterDF):
	'''
	Some columns of master need to be separated. For example, Rush-Yds-TDs will be split into 3 columns.
	'''
	ppCSV = masterDF.copy(deep=True)

	ppCSV[['homeCurrWins', 'homeCurrLosses', 'homeCurrTies']] = ppCSV['homeRecord'].str.split('-', expand=True)
	ppCSV[['awayCurrWins', 'awayCurrLoses', 'awayCurrTies']] = ppCSV['awayRecord'].str.split('-', expand=True)
	ppCSV['homeCurrTies'].fillna(0, inplace=True)
	ppCSV['awayCurrTies'].fillna(0, inplace=True)
	ppCSV = ppCSV.drop('homeRecord', axis=1)
	ppCSV = ppCSV.drop('awayRecord', axis=1)

	# Split Rush-Yds-TDs column into 3 different columns.
	# Note since negative-sign ('-') is used for negative numbers and delimiters, we replace the negative integer with
	# the 'neg' string, to be replaced with negative sign later after delimitation split
	ppCSV['away Rush-Yds-TDs'] = ppCSV['away Rush-Yds-TDs'].str.replace('--', '-neg')
	ppCSV['home Rush-Yds-TDs'] = ppCSV['home Rush-Yds-TDs'].str.replace('--', '-neg')
	ppCSV[['awayRushes', 'awayRushYds', 'awayRushTDs']] = ppCSV['away Rush-Yds-TDs'].str.split('-', expand=True)
	ppCSV[['homeRushes', 'homeRushYds', 'homeRushTDs']] = ppCSV['home Rush-Yds-TDs'].str.split('-', expand=True)
	ppCSV['awayRushYds'] = ppCSV['awayRushYds'].str.replace('neg', '-')
	ppCSV['homeRushYds'] = ppCSV['homeRushYds'].str.replace('neg', '-')
	ppCSV = ppCSV.drop('away Rush-Yds-TDs', axis=1)
	ppCSV = ppCSV.drop('home Rush-Yds-TDs', axis=1)

	ppCSV['away Cmp-Att-Yd-TD-INT'] = ppCSV['away Cmp-Att-Yd-TD-INT'].str.replace('--', '-neg')
	ppCSV['home Cmp-Att-Yd-TD-INT'] = ppCSV['home Cmp-Att-Yd-TD-INT'].str.replace('--', '-neg')
	ppCSV[['awayPassCmp', 'awayPassAtt', 'awayPassYds', 'awayPassTDs', 'awayPassInts']] = ppCSV['away Cmp-Att-Yd-TD-INT'].str.split('-', expand=True)
	ppCSV[['homePassCmp', 'homePassAtt', 'homePassYds', 'homePassTDs', 'homePassInts']] = ppCSV['home Cmp-Att-Yd-TD-INT'].str.split('-', expand=True)
	ppCSV['awayPassYds'] = ppCSV['awayPassYds'].str.replace('neg', '-')
	ppCSV['homePassYds'] = ppCSV['homePassYds'].str.replace('neg', '-')
	ppCSV = ppCSV.drop('away Cmp-Att-Yd-TD-INT', axis=1)
	ppCSV = ppCSV.drop('home Cmp-Att-Yd-TD-INT', axis=1)

	ppCSV[['awaySacks', 'awaySackYds']] = ppCSV['away Sacked-Yards'].str.split('-', expand=True)
	ppCSV[['homeSacks', 'homeSackYds']] = ppCSV['home Sacked-Yards'].str.split('-', expand=True)
	ppCSV = ppCSV.drop('away Sacked-Yards', axis=1)
	ppCSV = ppCSV.drop('home Sacked-Yards', axis=1)

	ppCSV[['awayFumbles', 'awayFumblesLost']] = ppCSV['away Fumbles-Lost'].str.split('-', expand=True)
	ppCSV[['homeFumbles', 'homeFumblesLost']] = ppCSV['home Fumbles-Lost'].str.split('-', expand=True)
	ppCSV = ppCSV.drop('away Fumbles-Lost', axis=1)
	ppCSV = ppCSV.drop('home Fumbles-Lost', axis=1)

	ppCSV[['awayPenalties', 'awayPenaltyYds']] = ppCSV['away Penalties-Yards'].str.split('-', expand=True)
	ppCSV[['homePenalties', 'homePenaltyYds']] = ppCSV['home Penalties-Yards'].str.split('-', expand=True)
	ppCSV = ppCSV.drop('away Penalties-Yards', axis=1)
	ppCSV = ppCSV.drop('home Penalties-Yards', axis=1)

	return ppCSV


if __name__ == '__main__':
	# # STEP 1: Scrape all boxscores, save to individual csv files
	# # Scrape boxscores for years
	# for year in range(1995,2020):
	# 	try:
	# 		dataDF = extractAllBoxscores(year)
	# 		dataDF.to_csv('boxscore_data_{}.csv'.format(year))
	# 	except:
	# 		print('Failed to compute all of {}'.format(year))
	# 		continue

	# # STEP 2: Concatenate all individual csv files into one master file
	# Concatenate all boxscore CSVs and write to file
	# masterFile = concatBoxscoreCSVs(os.listdir())
	# masterFile.to_csv('master_boxscore_data_1970_2019.csv')

	# # STEP 3: Post-process master file to fit format for network input (Each column holds one stat)
	masterDF = pd.read_csv('master_boxscore_data_1970_2019.csv', index_col=0, dtype=str)
	ppCSV = postProcessColumns(masterDF)
	ppCSV.to_csv('pp_master_boxscore_data_1970_2019.csv')


