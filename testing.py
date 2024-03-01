import yfinance as yf
import pandas as pd
import finplot as fplt
import requests
from lxml import html

pd.set_option('future.no_silent_downcasting', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


import requests_cache
session = requests_cache.CachedSession('yf.cache')
session.headers['User-agent'] = 'my-program/1.0'

ticker = "AAPL, NKE"

print([ticker][0])


# targets.append(babel.numbers.format_currency(decimal.Decimal(target), "USD"))

# msft = yf.Ticker("MSFT")
# df = msft.history(period="1d")

# df_output = df["Close"].iloc[len(df) - 400:]
# print(df_output)

# fplt.candlestick_ochl(df[['Open','Close','High','Low']])
# fplt.show()

# industry_search = "technology"
# url = requests.get("https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_technology/")
# print(url)
# url = requests.get("https://www.marketwatch.com/investing/stocks/tech-stocks")
# print(url)
# info = html.fromstring(url.content)
# printing = info.xpath("/html/body/main/div[1]/div[1]/div[3]/div/div/div[2]/div/table/tbody/tr[1]/td[1]/div[1]/a/text()")
# print(printing)

# url = requests.get("https://finance.yahoo.com/most-active/")
# info = html.fromstring(url.content)
# print(info.xpath("/html/body/div[1]/div/div/div[1]/div/div[2]/div/div/div[6]/div/div/section/div/div[2]/div[1]/table/tbody/tr[1]/td[1]/a/text()"))

# ticker = ["qqq"]
# links = ['https://www.marketwatch.com/investing/stock/{}/analystestimates'.format(ticker.lower()) for ticker in ticker]
# print(links)
