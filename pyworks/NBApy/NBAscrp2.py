# パッケージ読み込み
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os

os.mkdir("../Desktop/NBA")

# 収集する年度のリスト
year_list = [i+2007 for i in range(10)]

# urlを格納するリストの初期化
url_list = [0]*len(year_list)

# 各年ごとのチーム名を格納するリストの初期化
team_list = [0]*len(year_list)

# スクレイピング先のもとのサイト
url_original = "http://basketball.realgm.com/nba/teams/Atlanta-Hawks/1/Stats/2016/Averages/All/player/All/desc/1/Regular_Season"

# urlを分割，対象年度を挿入
url_original_split = url_original.split("/")

for i in range(len(year_list)):
    url_original_split[8] = str(year_list[i])

    # 年度を挿入後，urlを連結
    url = '/'.join(url_original_split)

    # urlを読み込み
    html = urlopen(url)
    bsObj = BeautifulSoup(html)

    # パース
    page_navi = bsObj.findAll("div",{"class":"page-navigation open"})[0]

    
    # urlリストの子リストの初期化
    url_list_child = []

    # チーム名リストの子リストの初期化
    team_list_child = []
    
    # チームごとのurlを取得
    teams = page_navi.findAll("select")[1]
    for team in teams.findAll("option"):
        # print(team['value'])
        url_list_child.append(team['value'])
        
        team_list_child.append(team.get_text().replace(" ", "_"))

    # url_list[i] = url_list_child
    # team_list[i] = team_list_child

    # 年度ごとのディレクトリ作成
    os.mkdir("../Desktop/NBA/"+str(year_list[i]))

    # 対象年度のチーム名ファイルを作成，オープン    
    f2 = open("../Desktop/NBA/"+str(year_list[i])+"/team.csv", "w")

    # 対象年度においてチームごとにスクレイピング
    for j in [k+1 for k in range(30)]:

        
        # 対象年度．チームのurlを取得
        team_url = "http://basketball.realgm.com"+url_list_child[j]

        html_team = urlopen(team_url)
        bsObj_team = BeautifulSoup(html_team)

        # 対チームのデータテーブル領域取得
        try:
            stats1 = bsObj_team.findAll("table")[2]
        except IndexError as e:
            print(team_url + " : doesn't have the table!")
            continue


        # 年度ディレクトリにおいてチームのファイルを作成・オープン   
        f = open("../Desktop/NBA/"+str(year_list[i])+"/"+team_list_child[j]+".csv", "w")

        # データテーブルのヘッダーを入手，表示
        for head in stats1.findAll("th"):
            f.write(head.get_text()+",")

        f.write("\n")

        # データテーブルの内容を入手・書き込み    
        for vs_team in stats1.tbody.findAll("tr"):

            for vs_team_stat in vs_team.findAll("td"):
                f.write(vs_team_stat.get_text()+",")

            f.write("\n")

        f.close()

        f2.write(team_list_child[j]+",")
        f2.write("\n")

    f2.close()


        

















