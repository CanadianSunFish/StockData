import yfinance as yf
import pandas as pd


class Stock:
    def __init__(self, tkr, price, date):
        self.tkr = tkr
        self.price = price
        self.date = date


aapl = Stock("aapl", 0, 0)
aapl.tkr = yf.Ticker("AAPL")
aapl.price = aapl.tkr.history(period="1mo")

print(aapl.price[["Open", "Close"]])
