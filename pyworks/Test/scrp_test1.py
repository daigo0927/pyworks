# パッケージ読み込み
import lxml.html

from selenium import webdriver


# スクレイピング先のサイト(今回はゴリラ)
target_url = 'http://www.s-arcana.co.jp/blog/2016/03/24/3051'

# JSを回避
driver = webdriver.PhantomJS()

driver.get(target_url)

root = lxml.html.fromstring(driver.page_source)

# 指定したURL先のどの部分を抜き出すかを指定
links = root.cssselect('#content h1')

for link in links:
    # ファイル書き込み
    content = str(link.text)

    text_write = open('write.txt', 'w')

    text_write.write(content + '\n')

    text_write.close()
    
    # ターミナルへの出力テスト
    print(content)
