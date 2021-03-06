# read package
from urllib.request import urlopen
from bs4 import BeautifulSoup

# list of years
year_list = [2016-i for i in range(10)]

url_original = "http://www.landofbasketball.com/results/2015_2016_scores_full.htm"

# split original url
url_original_split = url_original.split("/")

url_split = url_original_split

for i in range(len(year_list)):
    url_split[-1] = str(year_list[i]-1)+"_"+str(year_list[i])+"_scores_full.htm"

    url = '/'.join(url_split)

    print(url)

    html = urlopen(url)
    bsObj = BeautifulSoup(html)

    main = bsObj.findAll("main",{"class":"main-content"})[0]
    result_table = main.find("table")

    # for j in range(len(result_table.findAll("tr")))

    result_body = result_table.findAll("tr")

    
    # print(result_body[3]['class']) # >> ['t-enc-5', 'a-left']
    # print(result_body[3].get_text())

    '''
    # print(result_body[4]['style']) # >> vertical-align:top
    match = result_body[4].div.findAll("div", recursive=False)
    team1 = match[0].findAll("div", recursive=False)
    team1_name = team1[0]
    team1_points = team1[1]
    print(team1_name.get_text().strip())
    print(team1_points.get_text().strip())
    state = match[2]
    print(state.get_text().strip())
    '''


    f = open("../Desktop/NBA/"+str(year_list[i])+"/regular.csv", "w")
    
    for result in result_body[3:]:

        try:
            if result['class'][0] == 't-enc-5':
                date = result.get_text().strip()
                date = date.replace(",", "")
        except KeyError as e:
            pass

        try:
            if result['style'] == 'vertical-align:top':
                
                match = result.div.findAll("div", recursive=False)
                
                team1 = match[0].findAll("div", recursive=False)
                team1_name = team1[0].get_text().strip()
                team1_points = team1[1].get_text().strip()
                
                team2 = match[1].findAll("div", recursive=False)
                team2_name = team2[0].get_text().strip()
                team2_points = team2[1].get_text().strip()
                
                state = match[2].get_text().strip()
                
                # print(date+","+team1_name+","+team1_points+team2_name+","+team2_points+","+state)
                f.write(date+",")
                f.write(team1_name+",")
                f.write(team1_points)
                f.write(team2_name+",")
                f.write(team2_points+",")
                f.write(state)
                f.write("\n")
                
        except KeyError as e:
            pass

        try:
            if result['valign'] == 'middle':
                # print('playoffs start!')
                f.close()
                f = open("../Desktop/NBA/"+str(year_list[i])+"/playoffs.csv", "w")
                
        except KeyError as e:
            pass
        
    f.close()
