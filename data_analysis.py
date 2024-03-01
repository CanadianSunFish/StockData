import time
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import constants as c
import finplot as fplt
import matplotlib.pyplot as plt
from typing import Optional, List

import requests_cache
session = requests_cache.CachedSession('yf.cache')
session.headers['User-agent'] = 'my-program/1.0'

import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

class Analysis():

    def __init__(self, ticker=None, stock_list=None, end=c.today, include_nasqaq=True):
        if ticker is not None:
            self.ticker = ticker.upper()
            self.yf_ticker = yf.Ticker(ticker, session=session)
        if stock_list is not None:
            if(len(stock_list) > 10):
                raise ValueError("Stock list cannot be longer than 10 elements.")
            if(len(stock_list) < 2):
                raise ValueError("Stock list cannot be shorter than 2 elements.")
        self.stock_list = stock_list
        self.end = end
        self.include_nasdaq = include_nasqaq

    def get_distribution(self):
        self.df = self.yf_ticker.history(start="2001-01-01", end=self.end, interval="1d")
        open_change = self.df["Open"].pct_change()
        close_change = self.df["Close"].pct_change()
        volume = self.df["Volume"]
        volume_change = self.df["Volume"].pct_change()

        # sns.kdeplot(open_change)
        # sns.kdeplot(close_change)
        # plt.show()

        # sns.kdeplot(volume_change)
        # plt.show()

        # sns.kdeplot(volume)
        # plt.show()

    def get_correlation(self):
        if self.stock_list is None:
            return
        df_title = str(self.stock_list[0])
        if self.include_nasdaq == True:
            self.stock_list.insert(0, "^IXIC")
            df_title = "NASDAQ"
        ticker = yf.Ticker(self.stock_list[0], session=session)
        df = ticker.history(start='2010-01-01', end=self.end)
        df[df_title] = df['Close']

        for i in self.stock_list[1:]:
            ticker = yf.Ticker(i, session=session)
            stock_history = ticker.history(start='2010-01-01', end=self.end)
            df[i] = stock_history['Close']

        df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis=1)

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes()
        ax.set_facecolor('#faf0e6')
        fig.patch.set_facecolor('#faf0e6')
        cor = df.corr()
        sns.heatmap(cor, annot=True, cmap="Blues")
        plt.savefig('heat_map.jpg')
        # plt.show()


analysis = Analysis(stock_list=["AAPL", "NKE", "SCHL", "AMZN", "TSLA", "DOC", "NVDA", "AI"], include_nasqaq=True)
analysis.get_correlation()

end_time = time.time()
execution_time = end_time - start_time
print(f"The code ran in: {execution_time}s")
