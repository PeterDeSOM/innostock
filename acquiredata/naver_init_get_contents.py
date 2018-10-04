import bs4
import time
import requests
import os


class NavernewsInit:
    _NEWS_URL = 'http://news.naver.com/main/list.nhn?sid2=258&sid1=101&mid=shm&mode=LS2D'
    _NEWS_URL_DIR = './URLs/News/Naver/'
    _NEWS_TXT_DIR = './TXTs/News/Naver/'
    _NEWS_CONT_DIR = './Contents/News/Naver/'
    _NEWS_DATE = None

    ARTICLE_DELIMITER = "@@@@@ ARTICLE DELMITER @@@@\n"
    TITLE_START_PAT = '<h3 class="font1">'
    TITLE_END_PAT = '</h3>'
    DATE_TIME_START_PAT = u'기사입력 <span class="t11">'
    BODY_START_PAT = '<div class="article_body">'
    BODY_END_PAT = '<div class="article_footer">'
    TIDYUP_START_PAT = '<div class="article_footer">'

    def open_html_file(self):
        """
        HTML 기사 파일을 열어서 파일 객체를 돌려준다.
        :param html_file_name:
        :return:
        """

        htmlfile = self._NEWS_TXT_DIR + self._NEWS_DATE + '.txt'
        html_file = open(htmlfile, "r", encoding="utf-8")

        return html_file

    def create_text_file(self):
        """
        텍스트 기사 파일을 만들어 파일 객체를 돌려준다.
        :param text_file_name:
        :return:
        """

        if not os.path.exists(self._NEWS_CONT_DIR):
            os.makedirs(self._NEWS_CONT_DIR, mode=777)

        contfile = self._NEWS_CONT_DIR + self._NEWS_DATE + '.txt'
        content_file = open(contfile, "w", encoding="utf-8")

        return content_file

    def read_html_article(self, html_file):
        """
        HTML 파일에서 기사 하나를 읽어서 돌려준다.
        :param html_file:
        :return:
        """

        lines = []
        for line in html_file:
            if line.startswith(self.ARTICLE_DELIMITER):
                html_text = "".join(lines).strip()
                return html_text
            lines.append(line)

        return None

    def ext_title(self, html_text):
        """
        HTML 기사에서 제목을 추출하여 돌려준다.
        :param html_text:
        :return:
        """
        p = html_text.find(self.TITLE_START_PAT)
        q = html_text.find(self.TITLE_END_PAT)
        title = html_text[p + len(self.TITLE_START_PAT):q]
        title = title.strip()

        return title


    def ext_date_time(self, html_text):
        """
        HTML 기사에서 날짜와 시간을 추출하여 돌려준다.
        :param html_text:
        :return:
        """
        start_p = html_text.find(self.DATE_TIME_START_PAT)+len(self.DATE_TIME_START_PAT)
        end_p = start_p + 10
        date_time = html_text[start_p:end_p]
        date_time = date_time.strip()

        return date_time

    def strip_html(self, html_body):
        """
        HTML 본문에서 HTML 태그를 제거하고 돌려준다.
        :param html_body:
        :return:
        """
        page = bs4.BeautifulSoup(html_body, "html.parser")
        body = page.text

        return body

    def tidyup(self, body):
        """
        본문에서 필요없는 부분을 자르고 돌려준다.
        :param body:
        :return:
        """

        p = body.find(self.TIDYUP_START_PAT)
        body = body[:p]
        body = body.strip()

        return body

    def ext_body(self, html_text):
        """
        HTML 기사에서 본문을 추출하여 돌려준다.
        :param html_text:
        :return:
        """

        p = html_text.find(self.BODY_START_PAT)
        q = html_text.find(self.BODY_END_PAT)
        html_body = html_text[p + len(self.BODY_START_PAT):q]
        html_body = html_body.replace("<br />","\n")
        html_body = html_body.strip()
        body = self.strip_html(html_body)
        body = self.tidyup(body)

        return body

    def write_article(self, text_file, title, date_time, body):
        """
        텍스트 파일에 항목이 구분된 기사를 출력한다.
        :param text_file:
        :param title:
        :param date_time:
        :param body:
        :return:
        """

        text_file.write("{}\n".format(title))
        text_file.write("{}\n".format(date_time))
        text_file.write("{}\n".format(body))
        text_file.write("{}\n".format(self.ARTICLE_DELIMITER))

    def run(self, newsdate):
        """
        네이트 뉴스 기사 HTML에서 순수 텍스트 기사를 추출한다.
        :return:
        """

        self._NEWS_DATE = newsdate

        html_file = self.open_html_file()
        text_file = self.create_text_file()

        while True:

            html_text = self.read_html_article(html_file)

            if not html_text:
                break

            title = self.ext_title(html_text)
            date_time = self.ext_date_time(html_text)
            body = self.ext_body(html_text)
            self.write_article(text_file, title, date_time, body)

            print("'%s, %s' is processed." % (title, date_time))

        html_file.close()
        text_file.close()