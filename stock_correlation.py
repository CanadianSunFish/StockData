import time
import requests
import numpy as np
import pandas as pd
import constants as c
import yfinance as yf
import seaborn as sns
import data_collection as data
import matplotlib.pyplot as plt

from lxml import html
from datetime import date
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class Correlation():

    def __init__(self, stocks=["AAPL"], date=c.today)
        if isinstance(stocks, list):
            self.stocks = stocks
        else:
            self.stocks = [stocks]
        self.date = date

    def get_correlation(self):



today = date.today()
today = today.strftime("%Y-%m-%d")
stock = 'MCO'

ticker = yf.Ticker("^IXIC")
df = ticker.history(start='2010-01-01', end=today)
df['NASDAQ'] = df['Close']

stock_list = data.DataScraping().get_volume_leaders()

j = 1
for i in stock_list:
    if j == 10:
        break
    ticker = yf.Ticker(i)
    df1 = ticker.history(start='2010-01-01', end=today)
    df[i] = df1['Close']
    j += 1

df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis=1)
print(df)

fig = plt.figure(figsize=(10, 8))
ax = plt.axes()
ax.set_facecolor('#faf0e6')
fig.patch.set_facecolor('#faf0e6')
cor = df.corr()
sns.heatmap(cor, annot=True, cmap="Blues")
plt.savefig('heat_map.jpg')
plt.show()
