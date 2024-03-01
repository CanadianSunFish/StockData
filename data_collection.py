import time
import decimal
import requests
import warnings
import pandas as pd
import babel.numbers
import yfinance as yf
import constants as c
from lxml import html
from functools import lru_cache

import requests_cache
session = requests_cache.CachedSession('yf.cache')
session.headers['User-agent'] = 'my-program/1.0'

warnings.filterwarnings('ignore')
start_time = time.time()

# Using wrapper to check method arguments
def check(allowed, type_check):
    def wrapper(method):
        def inner(self, *args, **kwargs):
            if(type_check == "market"):
                if self.market not in allowed:
                    raise ValueError("Method not available for market '{}'. Allowed markets: {}".format(self.market, ", ".join(allowed)))
            return method(self, *args, **kwargs)
        return inner
    return wrapper


class StockData():

    def __init__(self, ticker="AAPL", market="stock", end=c.today):
        self.ticker = ticker.upper()
        self.yf_ticker = yf.Ticker(ticker, session=session)
        self.market = market
        self.end = end
        self.df = self.yf_ticker.history(start="2001-01-01", end=self.end, interval="1d")
        self.price = yf.download(self.ticker, period="1d")["Close"].iloc[0].round(2)
        self.info = self.yf_ticker.info

    def __str__(self):
        pricing = babel.numbers.format_currency(decimal.Decimal(self.price), "USD")
        return f"Ticker: {self.ticker}, Price: {pricing}"

    def get_live_data(self):
        pass

    def set_ticker(self, new_ticker):
        self.ticker = new_ticker.upper()
        self.yf_ticker = yf.Ticker(self.ticker)


class DataScraping():

    def __init__(self, tickers=["AAPL"], market="stock"):
        if isinstance(tickers, list):
            self.tickers = tickers
        else:
            self.tickers = [tickers]
        self.market = market.lower()
        self.sectors = None

        # Changing scrape site based on market type
        if (self.market == "stock"):
            self.links = ['https://www.marketwatch.com/investing/stock/{}/analystestimates'.format(ticker.lower()) for ticker in self.tickers]
        if (self.market == "fund"):
            self.links = ['https://www.marketwatch.com/investing/fund/{}/holdings?mod=mw_quote_tab'.format(ticker.lower()) for ticker in self.tickers]

    """
    Gets a list of every major stock sector.

    :returns: list of major stock sectors
    :rtyper: list of strings
    """
    def get_sectors(self):
        if self.sectors is None:
            url = requests.get("https://finance.yahoo.com/sectors/")
            info = html.fromstring(url.content)
            self.sectors= [info.xpath(f"/html/body/div[1]/main/section/section/section/article/section[1]/section[2]/div/div/div[1]/div/div[2]/table/tbody/tr[{i}]/td[1]/text()")[0] for i in range(2,13)]
        return self.sectors


    # TODO: Find way of getting all stocks. Currently recieving 404 for some sectors.
    """
    Gets a list of the top stocks from every sector returned from the get_sectors function.

    :returns: top stocks of every sector
    :rtype: dataframe
    """
    def get_industry_leaders(self):
        df = pd.DataFrame()
        for sector in self.get_sectors():
            sector_search = sector.replace(" ", "-").lower()

            url = requests.get(f"https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_{sector_search}")
            info = html.fromstring(url.content)
            sector_list = [info.xpath(f"/html/body/div[1]/div/div/div[1]/div/div[2]/div/div/div[6]/div/div/section/div/div[2]/div[1]/table/tbody/tr[{i}]/td[1]/a/text()") for i in range (1, 11)]

            df[sector] = sector_list

        return df

    """
    Gets stocks that had the highest trade volume for today.

    :return: list of stocks
    :rtype: list of strings
    """
    def get_volume_leaders(self):
        url = requests.get("https://finance.yahoo.com/most-active/")
        info = html.fromstring(url.content)
        leaders = [info.xpath(f"/html/body/div[1]/div/div/div[1]/div/div[2]/div/div/div[6]/div/div/section/div/div[2]/div[1]/table/tbody/tr[{i}]/td[1]/a/text()")[0] for i in range(1, 26)]
        return leaders

    """
    Gets most common analyst rating for given stock. This is averaged across all ratings. If a
    list of stocks is passed then a dataframe will be returned, otherwise just a string.

    :return: rating (overweight, underweight, buy, sell, hold)
    :rtype: string or dataframe
    :raises ValueError: if self.market is not "stock"
    """
    @check(["stock"], "market")
    def get_stock_rating(self):

        if len(self.tickers) == 1:
            url = requests.get(self.links[0])
            info = html.fromstring(url.content)
            rating = info.xpath('/html/body/div[3]/div[6]/div[1]/div[1]/div/table/tbody/tr[1]/td[2]/text()')[0]
            return rating

        df = pd.DataFrame()
        ratings = []
        df["Stocks"] = self.tickers
        for link in self.links:
            url = requests.get(link)
            info = html.fromstring(url.content)
            rating = info.xpath('/html/body/div[3]/div[6]/div[1]/div[1]/div/table/tbody/tr[1]/td[2]/text()')[0]
            ratings.append(rating)

        df["Rating"] = ratings
        return df

    """
    Gets most common analyst price target for given stock. This is averaged across all targets.
    If a list of stocks is passed then a dataframe will be returned, otherwise just an int. The
    dataframe will also contain the stocks current price.

    :return: price target
    :rtype: int or dataframe
    :raises ValueError: if self.market is not "stock"
    """
    @check(["stock"], "market")
    def get_stock_target(self):

        if (len(self.links) == 1):
            url = requests.get(self.links[0])
            info = html.fromstring(url.content)
            target = info.xpath('/html/body/div[3]/div[6]/div[1]/div[1]/div/table/tbody/tr[2]/td[2]/text()')[0]
            return target

        df = pd.DataFrame()
        df["Stocks"] = self.tickers

        # TODO: Figure out time complexity for requesting html twice in comprehension vs once in for loop
        targets = []
        current_price = []
        for link in self.links:
            url = requests.get(link)
            info = html.fromstring(url.content)
            target = info.xpath('/html/body/div[3]/div[6]/div[1]/div[1]/div/table/tbody/tr[2]/td[2]/text()')[0]
            targets.append(target)
            price  = info.xpath('/html/body/div[3]/div[1]/div[3]/div/div[2]/h2/bg-quote/text()')[0]
            current_price.append(price)


        df["Target"] = targets
        df["Current Price"] = current_price
        return df

    """
    Gets a list of all holdings maintained by the given etf.

    :return: fund holdings
    :rtype: list of strings
    """
    @check(["fund"], "market")
    def get_fund_holdings(self):
        url = requests.get(self.links[0])
        info = html.fromstring(url.content)

        target = [info.xpath(f"/html/body/div[3]/div[6]/div[3]/div/table/tbody/tr[{i}]/td[2]/text()")[0] for i in range(1, 50) if len(info.xpath(f"/html/body/div[3]/div[6]/div[3]/div/table/tbody/tr[{i}]/td[2]/text()"))>0]
        return target

data = DataScraping()
print(data.get_volume_leaders())

end_time = time.time()
execution_time = end_time - start_time
print(f"The code ran in: {execution_time}s")
