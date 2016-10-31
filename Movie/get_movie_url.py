
# read package
import time
import os
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver

file = open('movie_url.csv', 'w')
file.write('title'+','+'ref')
file.write('\n')




url_original = 'http://movies.yahoo.co.jp/movie/?type=&roadshow_flg=0&roadshow_from=&img_type=2&query=&genre=&award=&sort=-contribution&page='
url_head = 'http://movies.yahoo.co.jp/movie'

i = 1

while i <= 10:
    url = url_original+str(i)

    try:
        html = urlopen(url)
        time.sleep(3)
    except HTTPError as e:
        print('HTTP error ... , try again')
        continue

    print('scraping page '+str(i))
        
        
    bsObj = BeautifulSoup(html, 'html.parser')

    article = bsObj.find('article')
    section_list = article.find('div', {'id':'lst'}).ul
    
    movie_list = section_list.findAll('li', recursive=False)

    for movie in movie_list:
    
        # print(movie.a['title'])
        file.write(movie.a['title']+',')
        # print(movie.a['href'])
        file.write(movie.a['href'])
        file.write('\n')

    i += 1



file.close()
