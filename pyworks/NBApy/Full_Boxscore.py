# read package
import time
import re
import os
import datetime
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver


if not os.path.isdir('../../Desktop/NBA/2016/Regular'):
	os.mkdir("../../Desktop/NBA/2016/Regular")

if not os.path.isdir('../../Desktop/NBA/2016/Playoffs'):
        os.mkdir('../../Desktop/NBA/2016/Playoffs')


# file = open('../../Desktop/TEST/test.csv', 'w')
# file = open("../../Desktop/NBA/2016/score_p.csv", "w")
# file.write('\n')

# score_urls = open('../../Desktop/URLs/score_url.csv')
score_urls = open('../../Desktop/URLs/p_score_url.csv')

# scrape game score
score_url = score_urls.readlines()
# score_url = '/game/#!/0021500001'
score_urls.close()

url_head = 'http://stats.nba.com'

i=0

while i <= len(score_url)-1:

        # file_pass = '../../Desktop/NBA/2016/Regular/score'
        file_pass = '../../Desktop/NBA/2016/Playoffs/score'
        
        file = open(file_pass+str(i+1)+'_1.csv', 'w')
        
        # print(len(score_url))
        # print(score_url[1230].rstrip('\n'))
        # print(score_url[0].rstrip('\n'))
        url = url_head + score_url[i].rstrip('\n')+'/'
        # url = url_head + s.rstrip('\n') +'/'
        
        print(url + ' scrape start ...')

        start = time.time()
        driver = webdriver.PhantomJS()
        driver.get(url)
        time.sleep(3)
        elapsed_time = time.time() - start
        print('elapsed_time:{0}'.format(elapsed_time))

        pageSource = driver.page_source
        bsObj = BeautifulSoup(pageSource, 'html.parser')
        
        driver.close()

        stat_table = bsObj.find('div', {'class':'stat-table'})

        stat_team1 = stat_table.findAll('div', recursive=False)[-2]

        # team1 header : Player, min, ... ----------------------------------
        team1_head = stat_team1.find('thead')

        try:
                for h in team1_head.findAll('th'):
                        # print(h.get_text())
                        file.write(h.get_text()+',')
        except AttributeError as e:
                print('scraping error occured ,, try again')
                file.close()
                continue


        file.write('\n')

        # team1 name : Atlanta Hawks  **************************
        team1_name = stat_team1.find('div')
        # print(team1_name.get_text())
        file.write(team1_name.get_text()+',')

        # team1 sum score : ... ***************************
        team1_foot = stat_team1.find('tfoot')
        for f1 in team1_foot.findAll('td'):
                elem = re.search('-?\d+\:?\.?\d?', f1.get_text())
                if elem:
                        # print(elem.group())
                        file.write(elem.group()+',')
                else:
                        pass
			# print('none')
			# file.write('none,')

        file.write('\n')
                
        # team1 member score ---------------------------
        team1_body = stat_team1.find('tbody')
        # get player row
        for b1 in team1_body.findAll('tr'):

                # the player performed
                if len(b1.findAll('td'))>5:
                        # print('join')
                        
                        # get player element
                        for j in range(len(b1.findAll('td'))):
                                element = b1.findAll('td')[j]
                                
                                # first iterate : player name
                                if j==0:

                                        elem = re.search('\w+\s?\w+-?\w*', element.get_text())
                                        if elem:
                                                # print(elem.group())
                                                file.write(elem.group()+',')
                                        else:
                                                pass
                                        
                                # remain iterate : player score
                                else:
                                        elem = re.search('-?\d+\:?\.?\d*', element.get_text())
                                        if elem:
                                                # print(elem.group())
                                                file.write(elem.group()+',')
                                        else:
                                                pass
                                        # print('none')
                                        # file.write('none,')
                                        
                        file.write('\n')
                                        
                        
                # the player not performed
                else:
                        # print('not joined')
                       player = b1.find('td')
                       elem = re.search('\w+\s?\w+-?\w*', player.get_text())
                       if elem:
                               # print(elem.group())
                               file.write(elem.group()+',')
                       else:
                               pass

                       for j in range(20):
                               # print('NA')
                               file.write('NA,')

                       file.write('\n') 
                               
        file.close()

        file = open(file_pass+str(i+1)+'_2.csv', 'w')
        
        stat_team2 = stat_table.findAll('div', recursive=False)[-1]


        # team2 header : Player, min, ... ----------------------
        team2_head = stat_team2.find('thead')
        for h in team2_head.findAll('th'):
                # print(h.get_text())
                file.write(h.get_text()+',')

        file.write('\n')

	# team2 name : Boston Celtics **********************
        team2_name = stat_team2.find('div')
	# print(team2_name.get_text())
        file.write(team2_name.get_text()+',')

	# team2 sum score: ,,, ********************************
        team2_foot = stat_team2.find('tfoot')
        for f2 in team2_foot.findAll('td'):
                elem = re.search('-?\d+\:?\.?\d?', f2.get_text())
                if elem:
			# print(elem.group())
                        file.write(elem.group()+',')
                else:
                        pass
                # print('none')
                # file.write('none,')

        file.write('\n')

        # team2 member score ---------------------------
        team2_body = stat_team2.find('tbody')
        # get player row
        for b2 in team2_body.findAll('tr'):

                # the player performed
                if len(b2.findAll('td'))>5:
                        # print('join')
                        
                        # get player element
                        for j in range(len(b2.findAll('td'))):
                                element = b2.findAll('td')[j]
                                
                                # first iterate : player name
                                if j==0:
                                        # print(element.get_text())
                                        elem = re.search('\w+\s?\w+-?\w*', element.get_text())
                                        if elem:
                                                # print(elem.group())
                                                file.write(elem.group()+',')
                                        else:
                                                pass
                                        
                                # remain iterate : player score
                                else:
                                        elem = re.search('-?\d+\:?\.?\d*', element.get_text())
                                        if elem:
                                                # print(elem.group())
                                                file.write(elem.group()+',')
                                        else:
                                                pass
                                        # print('none')
                                        # file.write('none,')

                        file.write('\n')
                        
                # the player not performed
                else:
                        # print('not joined')
                       player = b2.find('td')
                       elem = re.search('\w+\s?\w+-?\w*', player.get_text())
                       if elem:
                               # print(elem.group())
                               file.write(elem.group()+',')
                       else:
                               pass

                       for j in range(20):
                               # print('NA')
                               file.write('NA,')

                       file.write('\n')

        file.close()
        i = i+1


