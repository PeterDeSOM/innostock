# Innostock

This repository contains source code for testing and validating neural network models before designing the production application.

---

## 📊 Web Crawler

For the Korean stock market, the **Korea Exchange (KRX)** and **DART (Data Analysis, Retrieval and Transfer System)** are the most reliable sources of market and corporate data.  
The crawler is responsible for continuously fetching and preprocessing daily datasets, which serve as inputs for model training and analysis.

### References
- [Korea Exchange (KRX)](http://www.krx.co.kr)  
- [DART (FSS)](https://dart.fss.or.kr)

### Data Requirements
The following historical and derived data points are required:

1. 10 years of daily trading history  
2. Open, Bid, Ask, and Close prices  
3. Daily trading volume  
4. 50-, 30-, and 20-day average daily volumes  
5. Price-to-Earnings (P/E) ratio  
6. Forward P/E (1 year)  
7. Earnings Per Share (EPS)  

> ⚠️ While basic trading data (prices, volumes) are straightforward to extract, advanced features require additional engineering. Examples include classifying price movements as positive/negative or deriving normalized mean-value indicators during high volatility.

---

## 🤖 Neural Network Models

The objective is to determine which architecture best predicts the next day’s stock price. Each model was tested independently:

- **CNN** (Convolutional Neural Networks)  
- **RNN** (including LSTM and GRU)  
- **NLP-based models** (for sentiment and corporate filings analysis)  
- **RL** (Reinforcement Learning: Actor-Critic, A2C, A3C)  

---

## 🛠 Tools & Technologies

- **Languages/Frameworks**: Python, R  
- **Libraries**: TensorFlow, Keras  
- **Database**: MariaDB  

---
