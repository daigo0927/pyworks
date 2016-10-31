# tutorial program

# read package
import os
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver

file = open('movie_url.csv', 'w')
file.write('title'+','+'ref')
file.write('\n')




url_original = 'http://movies.yahoo.co.jp/movie/?type=&roadshow_flg=0&roadshow_from=&img_type=2&query=&genre=&award=&sort=-contribution&page=1'
url_head = 'http://movies.yahoo.co.jp/movie'

url = url_original

# JavaScript version

'''
driver = webdriver.PhantomJS()
driver.get(url)

pageSource = driver.page_source
bsObj = BeautifulSoup(pageSource, 'html.parser')

driver.close()

article = bsObj.find('article')
section_list = article.find('div', {'id':'lst'}).ul

movie_list = section_list.findAll('li', recursive=False)

for movie in movie_list:
    print(movie.a['href'])
    print(movie.a['title'])
'''

html = urlopen(url)
bsObj = BeautifulSoup(html, 'html.parser')

article = bsObj.find('article')
section_list = article.find('div', {'id':'lst'}).ul

movie_list = section_list.findAll('li', recursive=False)

for movie in movie_list:
    
    print(movie.a['title'])
    file.write(movie.a['title']+',')
    print(movie.a['href'])
    file.write(movie.a['href'])
    file.write('\n')

file.close()
    

    

