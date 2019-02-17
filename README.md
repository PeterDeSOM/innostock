# Innostock

These source codes are used to test each of the neural net models before designing the app.

## Web crawler

KRX and DART are the great sources for Korean stock market data, hence the scraper bringing their daily trading data should be running in front of the analyzer. 

### References
- Korea Exchange http://www.krx.co.kr 
- Data Analysis, Retrieval and Transfer System https://dart.fss.or.kr

### Extracting data (What is the data should be prepared?)

1. Past 10 Years daily transding history
2. Open, Bid, Ask, Close price
3. Share Volume
4. 50, 30, 20 Day Avg. Daily Volume
5. P/E Ratio
7. Forward P/E (1y)
8. Earnings Per Share (EPS)

Most of the simple data, e.g., Open, Bid, Ask, Close price are easy to extract, but some can not. For instance, classifying meaning of the stock price fluctuation as positive or negative and finding a mean value point for the high differences of the stock price or its volume should be created out of nothing.

## Neural net models

I tried to find out which model can effectively predict the next day's stock price, that's why each of the models was tested one by one. Here is what I did.

- CNN
- RNN (LSTM / GRU)
- NLP
- RL (Actor Critic, A2C, A3C)

### Tools

- Python
- Tensorflow
- Keras
- MariaDB
- R
