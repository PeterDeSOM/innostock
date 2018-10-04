import time
import re
import requests
import random
import os


class NavernewsInit:
    _NEWS_URL = 'http://news.naver.com/main/list.nhn?sid2=258&sid1=101&mid=shm&mode=LS2D'
    _NEWS_URL_DIR = './URLs/News/Naver/'
    _NEWS_URL_FILENAME = None
    _NEWS_DATE = None

    def create_output_file(self):
        urlfile = self._NEWS_URL_DIR + self._NEWS_DATE + '.txt'
        if not os.path.exists(self._NEWS_URL_DIR):
            os.makedirs(self._NEWS_URL_DIR, mode=777)

        output_file = open(urlfile, "w", encoding="utf-8")

        return output_file

    def get_html(self, page_num):
        user_agent = "'Mozilla/5.0"
        headers ={"User-Agent" : user_agent}

        page_url = self._NEWS_URL + '&date=%s&page=%d' % (self._NEWS_DATE, page_num)

        response = requests.get(page_url, headers=headers)
        html = response.text

        return html

    def ext_news_article_urls(self, html):
        """
        주어진 html에서 기사 url을 추출하여 돌려준다.
        :param html:
        :return:
        """

        url_frags = re.findall('<a href="(.*?)"', html)
        news_article_urls=[]

        for url_frag in url_frags:
            if "sid1=101&sid2=258" in url_frag and "aid" in url_frag:
                news_article_urls.append(url_frag)
            else :
                continue

        return news_article_urls

    def write_news_article_urls(self, output_file, urls):
        """
        기사 URL들을 출력 파일에 기록한다.
        :param output_file:
        :param urls:
        :return:
        """
        for url in urls:
            print(url, file=output_file)

    def close_output_file(self, output_file):
        """
        출력파일을 닫는다.
        :param output_file:
        :return:
        """
        output_file.close()

    def run(self, newsdate):
        self._NEWS_DATE = newsdate

        # target_date = self.get_date()
        output_file = self.create_output_file()
        page_num = 1
        max_page_num = 10

        while True :
            html = self.get_html(page_num)

            if page_num>=max_page_num:
                break

            urls = self.ext_news_article_urls(html)
            if len(urls) == 0:
                break

            self.write_news_article_urls(output_file, urls)

            page_num+=1

            # time.sleep('{:.{prec}f}'.format(random.uniform(0.25, 2.00), prec=2))
            time.sleep(round(random.uniform(0.50, 3.00), 2))

            print(newsdate, " : Page No ", page_num-1)

        self.close_output_file(output_file)
