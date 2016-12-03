# read package
import time
import os
import datetime
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver

# scrape NBA 2015-16 season scores

start_regular_date = datetime.date(2015, 10, 27) # 2015/10/27
regular_date = start_regular_date
end_regular_date = datetime.date(2016, 4, 13) # 2016/4/13

start_playoffs_date = datetime.date(2016, 4, 16)
playoffs_date = start_playoffs_date
end_playoffs_date = datetime.date(2016, 6, 20)

# print(end_playoffs_date.strftime('%m/%d/%Y'))
# >> '6/20/2016'


if not os.path.isdir('../../Desktop/URLs'):    
    os.mkdir('../../Desktop/URLs')
    
file = open('../../Desktop/URLs/score_url.txt', 'w')


url_start = 'http://stats.nba.com/scores/#!/10/27/2015'

# games url in a day
url_scores = 'http://stats.nba.com/scores/#!/'
# url head
url_head = 'http://stats.nba.com'

# box socre urls : http://stats.nba.com/game/#!/0000XXXXX/

# scrape box score urls ***************************************

# box scoreのurlを格納するリストの初期化
score_url_list = []
s_num = 0 # score element number

# レギュラーシーズンのスコアurl取得　*********************


while regular_date <= end_regular_date:

    driver = webdriver.PhantomJS()

    url = url_scores + regular_date.strftime('%m/%d/%Y')
    driver.get(url)
    time.sleep(4)

    pageSource = driver.page_source
    bsObj = BeautifulSoup(pageSource, 'html.parser')

    driver.close()

    gameWindows = bsObj.find('div', {'data-ng-hide':'isLoading', 'class':'row'})
    games = gameWindows.findAll('div', recursive=False)

    for game in games :
        gameRef = game.find('div', {'class':'game-footer'})
        try:
            scoreRef = gameRef.find('a', {'class':'game-footer__bs'})
            # print(scoreRef['href'])
            score_url_list.append(scoreRef['href'])
            # print(score_url_list[s_num])
            # /game/#!/00000000XXXX
            # -> http://stats.nba.com/game/#!/0000XXXXX/
            file.write(scoreRef['href'])
            file.write('\n')
            s_num = s_num+1
            
        except AttributeError as e:
            print('No Game Scheduled')
            continue

    print(regular_date.strftime('%m/%d/%Y')+' game urls obtained')
    regular_date = regular_date + datetime.timedelta(days=1)

file.close()

# レギュラーシーズンのスコアのurl取得完了***********
    



# プレーオフのスコアのurl取得******************

pfile = open('../../Desktop/URLs/p_score_url.txt', 'w')

# box scoreのurlを格納するリストの初期化
p_score_url_list = []
p_s_num = 0 # score element number

while playoffs_date <= end_playoffs_date:

    driver = webdriver.PhantomJS()

    url = url_scores + playoffs_date.strftime('%m/%d/%Y')
    driver.get(url)
    time.sleep(4)

    pageSource = driver.page_source
    bsObj = BeautifulSoup(pageSource, 'html.parser')
    
    driver.close()

    gameWindows = bsObj.find('div', {'data-ng-hide':'isLoading', 'class':'row'})
    games = gameWindows.findAll('div', recursive=False)

    for game in games :
        gameRef = game.find('div', {'class':'game-footer'})
        try:
            scoreRef = gameRef.find('a', {'class':'game-footer__bs'})
            # print(scoreRef['href'])
            p_score_url_list.append(scoreRef['href'])
            # print(p_score_url_list[p_s_num])
            # /game/#!/00000000XXXX
            # -> http://stats.nba.com/game/#!/0000XXXXX/
            pfile.write(scoreRef['href'])
            pfile.write('\n')
            p_s_num = p_s_num+1
            
        except AttributeError as e:
            print(playoffs_date.strftime('%m/%d/%Y')+' No Game Scheduled')
            continue

    print(playoffs_date.strftime('%m/%d/%Y')+' game urls obtained')
    playoffs_date = playoffs_date + datetime.timedelta(days=1)


pfile.close()
# プレーオフのスコアのurl取得完了　**********************


# box score urls scrape completed ***************************


# box score scraping ****************************************


