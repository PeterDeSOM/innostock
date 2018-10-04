import time
import requests
import os
import random


class NavernewsInit:
    _NEWS_URL = 'http://news.naver.com/main/list.nhn?sid2=258&sid1=101&mid=shm&mode=LS2D'
    _NEWS_URL_DIR = './URLs/News/Naver/'
    _NEWS_URL_FILENAME = None
    _NEWS_TXT_DIR = './TXTs/News/Naver/'
    _NEWS_TXTL_FILENAME = None
    _NEWS_DATE = None

    def open_url_file(self):
        """
        URL 파일을 연다.
        :param url_file_name:
        :return:
        """
        urlfile = self._NEWS_URL_DIR + self._NEWS_DATE + '.txt'
        url_file = open(urlfile, "r", encoding ="utf-8")

        return url_file

    def create_output_file(self):
        """
        출력 파일을 생성한다.
        :param output_file_name:
        :return:
        """
        if not os.path.exists(self._NEWS_TXT_DIR):
            os.makedirs(self._NEWS_TXT_DIR, mode=777)

        txtfile = self._NEWS_TXT_DIR + self._NEWS_DATE + '.txt'
        output_file = open(txtfile, "w", encoding='utf-8')

        return output_file

    def gen_print_url(self, url_line):
        """
        주어진 기사 링크 URL로 부터 인쇄용 URL을 만들어 돌려준다.
        :param url_line:
        :return:
        """
        article_id = url_line[(len(url_line)-24):len(url_line)]
        print_url = "http://news.naver.com/main/tool/print.nhn?" + article_id

        return print_url

    def get_html(self, print_url) :
        """
        주어진 인쇄용 URL에 접근하여 HTML을 읽어서 돌려준다.
        :param print_url:
        :return:
        """
        user_agent = "'Mozilla/5.0"
        headers ={"User-Agent" : user_agent}

        response = requests.get(print_url, headers=headers)
        html = response.text

        return html

    def write_html(self, output_file, html):
        """
        주어진 HTML 텍스트를 출력 파일에 쓴다.
        :param output_file:
        :param html:
        :return:
        """

        output_file.write("{}\n".format(html))
        output_file.write("@@@@@ ARTICLE DELMITER @@@@\n")

    def close_output_file(self, output_file):
        """
        출력 파일을 닫는다.
        :param output_file:
        :return:
        """

        output_file.close()

    def close_url_file(self, url_file):
        """
        URL 파일을 닫는다.
        :param url_file:
        :return:
        """

        url_file.close()

    def run(self, newsdate):
        """
        네이버 뉴스기사를 수집한다.
        :return:
        """

        self._NEWS_DATE = newsdate

        url_file = self.open_url_file()
        output_file = self.create_output_file()

        for line in url_file:
            print_url = self.gen_print_url(line)
            html = self.get_html(print_url)
            self.write_html(output_file,html)

            print(newsdate, " : URL ", print_url)

            time.sleep(round(random.uniform(0.50, 3.00), 2))

        self.close_output_file(output_file)
        self.close_url_file(url_file)