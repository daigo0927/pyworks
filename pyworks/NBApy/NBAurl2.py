# パッケージ読み込み
from urllib.request import urlopen
from bs4 import BeautifulSoup


# 収集する年度のリスト
year_list = [i+2007 for i in range(2)]

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

    url_list[i] = url_list_child

    team_list[i] = team_list_child


for i in range(len(year_list)):
   for j in range(len(team_list[i])):
        print(team_list[i][j])
        print(url_list[i][j])


# for i in range(10):
#    print(url_list[0][i])
