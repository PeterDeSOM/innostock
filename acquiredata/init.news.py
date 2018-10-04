import pandas as pd
import konlpy

from acquiredata.naver_init_news import NavernewsInit as nviniturl
from acquiredata.naver_init_parsing_text import NavernewsInit as nvinittxt
from acquiredata.naver_init_get_contents import NavernewsInit as nvinitcnt
from datetime import datetime, timedelta
from databases import maria

datetoday = datetime.today()
daysbefore = (datetoday - datetime(2017, 1, 1)).days
for i in range (daysbefore, -1, -1):
    transDate = datetoday - timedelta(days=i)
    newsDate = transDate.strftime('%Y%m%d')

    """
    nvinitu = nviniturl()
    nvinitu.run(newsDate)

    nvinitt = nvinittxt()
    nvinitt.run(newsDate)
    """

    nvinitc = nvinitcnt()
    nvinitc.run(newsDate)

    # df_daytrans = pd.DataFrame()

    # mariadb = maria()
    # mariadb.insert("trans_daily", df_daytrans)

    print('%s daily news datas are processed.' % transDate.strftime('%Y-%m-%d'))

    break

# End for loop to Target date
