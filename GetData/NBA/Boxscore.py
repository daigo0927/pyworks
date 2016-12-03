# read package
import time
import re
import os
import datetime
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver


if not os.path.isdir('../../Desktop/NBA/2016'):
	os.mkdir("../../Desktop/NBA/2016")


# file = open('../../Desktop/TEST/test.csv', 'a')
file = open("../../Desktop/NBA/2016/score_p.csv", "w")
file.write('\n')

score_urls = open('../../Desktop/URLs/p_score_url.csv')

# scrape game score
score_url = score_urls.readlines()
# score_url = '/game/#!/0021500001'
score_urls.close()

url_head = 'http://stats.nba.com'

for s in score_url:

	url = url_head + s.rstrip('\n') +'/'

	print(url + ' scrape start ...')

	driver = webdriver.PhantomJS()
	driver.get(url)
	time.sleep(5)

	pageSource = driver.page_source
	bsObj = BeautifulSoup(pageSource, 'html.parser')

	driver.close()

	stat_table = bsObj.find('div', {'class':'stat-table'})

	stat_team1 = stat_table.findAll('div', recursive=False)[-2]


	# first iteration : write header 
	if url == 'http://stats.nba.com/game/#!/0021500001/' or url == 'http://stats.nba.com/game/#!/0041500111/':
		# header : FG, 3P, ... ***********************************
		team1_head = stat_team1.find('thead')
		for h in team1_head.findAll('th'):
			# print(h.get_text())
			file.write(h.get_text()+'1,')
		for h in team1_head.findAll('th'):
			# print(h.get_text())	
			file.write(h.get_text()+'2,')
		file.write('\n')

	# team1 name : Atlanta Hawks  **************************
	team1_name = stat_team1.find('div')
	# print(team1_name.get_text())
	file.write(team1_name.get_text()+',')

	# team1 score : ... ***************************
	team1_foot = stat_team1.find('tfoot')
	for f1 in team1_foot.findAll('td'):
		elem = re.search('-?\d+\.?\d?', f1.get_text())
		if elem:
			# print(elem.group())
			file.write(elem.group()+',')
		else:
			pass
			# print('none')
			# file.write('none,')

	stat_team2 = stat_table.findAll('div', recursive=False)[-1]

	# team2 name : Boston Celtics **********************
	team2_name = stat_team2.find('div')
	# print(team2_name.get_text())
	file.write(team2_name.get_text()+',')

	# team2 score: ,,, ********************************
	team2_foot = stat_team2.find('tfoot')
	for f2 in team2_foot.findAll('td'):
		elem = re.search('-?\d+\.?\d?', f2.get_text())
		if elem:
			# print(elem.group())
			file.write(elem.group()+',')
		else:
			pass
			# print('none')
			# file.write('none,')

	file.write('\n')

file.close()

