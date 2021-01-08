import requests
import bs4
import json
import datetime
import os
import csv
from random import randint
from math import pi
from time import sleep
import re
import pandas as pd

def extractBoxscoreLinks(year):
	'''
	The script reads the years/{yearNum}/games.htm page on PFR and iterates over the Week-by-Week table to extract
	 the boxscore links. Links are returned as follows:
	 :return: weekByWeek = {link1: {statName1: statVal1, statName2: statVal2}}
	 where
	 link1: suffix of link to PRF webpage (can access page by appending to http://www.pro-football-reference.com/)
	 statName#: stat extracted from Week-by-Week table
	 statVal#: value of stat
	'''
	# First, we iterate over the years/{yearNum}/games.htm page on PFR to extract all links to boxscores

	# Read games page for specified year from pro-football-reference with soup
	result = requests.get('http://www.pro-football-reference.com/years/{0}/games.htm'.format(year))
	content = re.sub(r'(?m)^\<!--.*\n?', '', result.content.decode('ISO-8859-1'))
	content = re.sub(r'(?m)^\-->.*\n?', '', content)
	soup = bs4.BeautifulSoup(content, 'html5lib')

	# Find table containing all games, extract rows from tbody of table
	gamesTable = soup.find('table', id='games')
	games = gamesTable.find('tbody')
	rows = games.find_all('tr')

	# Initialize dictionary which will contain info from weekByWeek table from webpage defined in result variable
	weekByWeek = {}

	# Iterate over each row in table
	for row in rows:

		# Extract hyperlink to game data.
		gameLink = row.find(attrs={'data-stat': 'boxscore_word'}).find('a')

		# If no hyperlink exists, skip this row (most likely a header row). Else save hyperlink address
		if gameLink == None:
			continue
		else:
			gameLink = gameLink['href']

		# Extract week, date, winner, and loser from row
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
		weekByWeek[gameLink] = {'date': date, 'week': week, 'home': home, 'away': away}
	return weekByWeek
def extractAllBoxscores(year):

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
	weekByWeek = extractBoxscoreLinks(year)

	# Initialize DataFrame where each row will be one game and each column contains different statistic
	dataDF = pd.DataFrame()

	# Iterate over all links, accessing boxscore webpage and extracting necessary data
	for gameLink, info in weekByWeek.items():
		# Initialize empty data dict which is used to store data from current gameLink on each iteration
		data = {}
		data['year'] = year
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
			if year >= 1999:

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

				# try:
				# 	pbpTable = soup.find('table', id='pbp')
				# 	pbpTable = pbpTable.find('tbody')
				# 	rows = pbpTable.find_all('tr')
				#
				#
				# 	for row in rows[1:]: #Skip first row since it is '1st Quarter' heading
				# 		play = row.find('td', id='detail').get_text()
				# 		if 'pass' not in play:
				# 			continue
				# 		else:
				# 			playList = play.split(' pass ')
				# 			playerName = playList[0]
				# 			if data['awayQB'] and data['homeQB']:
				# 				break
				#
				# 			if playerName in QBNames['awayQB']:
				# 				if not data['awayQB']:
				# 					data['awayQB'] = playerName
				# 			elif playerName in QBNames['homeQB']:
				# 				if not data['homeQB']:
				# 					data['homeQB'] = playerName
				# except:
				# 	print('No play by play for game :' + gameLink)
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


if __name__ == '__main__':
	for year in range(1970,1999):
		try:
			dataDF = extractAllBoxscores(year)
			dataDF.to_csv('boxscore_data_{}.csv'.format(year))
		except:
			print('Failed to compute all of {}'.format(year))
			continue
