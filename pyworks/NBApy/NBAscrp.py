# パッケージ読み込み
from urllib.request import urlopen
from bs4 import BeautifulSoup


# スクレイピング先のサイト
html = urlopen("http://basketball.realgm.com/nba/teams/Cleveland-Cavaliers/5/Stats/2016/Averages/All/player/All/desc/1/Regular_Season")
bsObj = BeautifulSoup(html)

f = open("write.txt","w")

# 対チームの統計データテーブル
stats1 = bsObj.findAll("table")[2]

# 統計データテーブルのヘッダーを入手，表示
for head in stats1.findAll("th"):
    f.write(head.get_text()+",")
    # print(head.get_text()+",")

f.write("\n")

# データテーブルの内容のうち最初のもの：対戦相手を入手，表示    
for vs_team in stats1.tbody.findAll("tr"):

    for vs_team_stat in vs_team.findAll("td"):
        f.write(vs_team_stat.get_text()+",")

    f.write("\n")

    # 内容のうち対戦相手ごとに全て入手，表示
    # print(vs_team.get_text())

f.close()

