# パッケージ読み込み
from urllib.request import urlopen
from bs4 import BeautifulSoup


# スクレイピング先のサイト
# html = urlopen("http://basketball.realgm.com/nba/teams/Cleveland-Cavaliers/5/Stats/2016/Averages/All/player/All/desc/1/Regular_Season")
html = urlopen("http://basketball.realgm.com/nba/teams/Atlanta-Hawks/1/Stats/2016/Averages/All/player/All/desc/1/Regular_Season")
bsObj = BeautifulSoup(html)


# パース
page_navi = bsObj.findAll("div",{"class":"page-navigation open"})[0]

# 年ごとのurlを取得
# years = page_navi.findAll("select")[0]
# for year in years.findAll("option"):
  #   print(year['value'])

# チームごとのurlを取得
teams = page_navi.findAll("select")[1]
for team in teams.findAll("option"):
    print(team['value'])


