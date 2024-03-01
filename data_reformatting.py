import numpy as np
import pandas as pd
import yfinance as yf
import constants as c
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


pd.set_option('future.no_silent_downcasting', True)

graphing_dates = [c.yesterday + timedelta(days=day) for day in range(80) if
                      (int((c.yesterday + timedelta(days=day)).isoweekday()) < 6) and (
                              (c.yesterday + timedelta(days=day)) not in c.holidays)]

def regression_data(df):
    df_open = df['Open']
    df_high = df['High']
    df_low = df['Low']
    df_close = df['Close']
    obv = df[['High', 'Volume']]

    df['Volume_Price'] = (df['Volume'] / df['Open']).rolling(20).mean()
    df['On_Balance_Volume'] = (np.sign(obv['High'].diff()) * obv['Volume']).cumsum()

    df['Simple_Moving_Average'] = df_open.rolling(20).mean()
    df['Exponential_Moving_Average'] = df_open.ewm(span=26, adjust=False).mean()
    df['Exponential_Moving_Average1'] = df_open.ewm(span=12, adjust=False).mean()
    df['MACD'] = (df['Exponential_Moving_Average'] - df['Exponential_Moving_Average1'])

    df['OBV_VolumePrice'] = ((df['Volume_Price'] + df['On_Balance_Volume']) / 2).rolling(20).mean()
    df['MACD_VolPrice'] = ((df['MACD'] + df['OBV_VolumePrice']) / 2).rolling(20).mean()

    df['OpenHighLow'] = ((df['Open'] + df['High'] + df['Low']) / 3)
    df['SMA_EMA'] = (df['Exponential_Moving_Average'] + df['Simple_Moving_Average']) / 2

    df = df[['MACD_VolPrice', 'OpenHighLow', 'SMA_EMA']]

    df = df.iloc[60:]

    return df


def prediction(df, output, days):
    # df[f'{output} Prediction'] = df[f'{output}']

    # X_predict = df.drop([f'{output} Prediction'], axis=1)
    # X_predict = np.array(X_predict[-days:])

    # y = df[f"{output}"]
    # X = df.drop([f"{output}"], axis=1)

    # X = np.array(X[:-days])
    # y = np.array(y[:-days])

    # regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=0)

    # scaler = StandardScaler()
    # scaler.fit(X.reshape(len(X), 7))
    # X = scaler.transform(X.reshape(len(X), 7))
    # X_predict = scaler.transform(X_predict.reshape(len(X_predict), 7))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    # regr.fit(X_train, y_train)

    # prediction = regr.predict(X_predict)

    # =======================
    df_output = df[f'{output}'].iloc[len(df) - 400:]

    df = regression_data(df)
    df = df.iloc[len(df) - 400:]

    df[f'{output} Prediction'] = df_output.shift(-days)
    X_predict = df.drop([f'{output} Prediction'], axis=1)
    X_predict = np.array(X_predict[-days:])

    X = df.drop([f'{output} Prediction'], axis=1)
    y = df[f'{output} Prediction']

    X = np.array(X[:-days])
    y = np.array(y[:-days])

    regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=0)

    scaler = StandardScaler()
    scaler.fit(X.reshape(len(X), 3))
    X = scaler.transform(X.reshape(len(X), 3))
    X_predict = scaler.transform(X_predict.reshape(len(X_predict), 3))

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    regr.fit(X_train, y_train)

    prediction = regr.predict(X_predict)

    return prediction


def data_cleaning(df):
    df = df.dropna()
    return df


aapl = yf.Ticker("GE")
symbol = "GE"
df = aapl.history(start="2001-01-01", end="2024-03-01", interval="1d")

days = 5
high = prediction(df, "High", days)
close = prediction(df, "Close", days)
# plt.plot(np.linspace(0, 5, 5), open, label="Open")

x = np.arange(1, days + 1)
y = high
y1 = close

X_Y_Spline = make_interp_spline(x, y)
X_Y_Spline1 = make_interp_spline(x, y1)

X_ = np.linspace(min(x), max(x), 250)
Y_ = X_Y_Spline(X_)
Y1_ = X_Y_Spline1(X_)

fig = plt.figure(figsize=(16, 8))
ax = plt.axes()

plt.plot(X_, Y_, label='High')
plt.plot(X_, Y1_, label='Close')
plt.fill_between(X_, Y_, Y1_, alpha=0.2)

def annot(x, y, type, ax=None):
    x_max = x[np.argmax(y)]
    x_min = x[np.argmin(y)]
    y_max = y.max()
    y_min = y.min()
    text = "High=${:.3f}".format(y_max)
    text1 = "Low=${:.3f}".format(y_min)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrow_props = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
                arrowprops=arrow_props, bbox=bbox_props, ha="right", va="top")
    if type == 'type_max':
        if x_max > 5:
            ax.annotate(text, xy=(x_max, y_max), xytext=(0.8, 0.96), **kw)
        else:
            ax.annotate(text, xy=(x_max, y_max), xytext=(0.2, 0.96), **kw)
    if type == 'type_min':
        if x_min > 5:
            ax.annotate(text1, xy=(x_min, y_min), xytext=(0.8, 0.06), **kw)
        else:
            ax.annotate(text1, xy=(x_min, y_min), xytext=(0.2, 0.06), **kw)

annot(X_, Y_, 'type_max')
annot(X_, Y1_, 'type_min')

plt.minorticks_on()
if high[0] > 10000:
    ax.set_ylim([min(y1) - (min(y1) * 0.01), max(y) + (max(y) * 0.01)])
    ax.yaxis.set_ticks(np.arange((min(y1) - min(y1) * 0.012), (max(y) + max(y) * 0.012), step=(y[4] * 0.012)))
elif high[0] > 1000:
    ax.set_ylim([min(y1) - (min(y1) * 0.01), max(y) + (max(y) * 0.01)])
    ax.yaxis.set_ticks(np.arange((min(y1) - min(y1) * 0.012), (max(y) + max(y) * 0.012), step=(y[4] * 0.014)))
elif high[0] > 250:
    ax.set_ylim([min(y1) - (min(y1) * 0.01), max(y) + (max(y) * 0.01)])
    ax.yaxis.set_ticks(np.arange((min(y1) - min(y1) * 0.012), (max(y) + max(y) * 0.012), step=(y[4] * 0.02)))
elif high[0] > 100:
    ax.set_ylim([min(y1) - (min(y1) * 0.01), max(y) + (max(y) * 0.01)])
    ax.yaxis.set_ticks(np.arange((min(y1) - min(y1) * 0.012), (max(y) + max(y) * 0.012), step=(y[4] * 0.018)))
elif high[0] > 50:
    ax.set_ylim([min(y1) - (min(y1) * 0.01), max(y) + (max(y) * 0.01)])
    ax.yaxis.set_ticks(np.arange((min(y1) - min(y1) * 0.012), (max(y) + max(y) * 0.012), step=(y[4] * 0.012)))
elif high[0] > 20:
    ax.set_ylim([min(y1) - (min(y1) * 0.01), max(y) + (max(y) * 0.01)])
    ax.yaxis.set_ticks(np.arange((min(y1) - min(y1) * 0.012), (max(y) + max(y) * 0.012), step=(y[4] * 0.012)))
else:
    ax.set_ylim([min(y1) - (min(y1) * 0.012), max(y) + (max(y) * 0.012)])
    ax.yaxis.set_ticks(np.arange(min(y1) - (min(y1) * 0.008), max(y) + (max(y) * 0.012), step=(y[4] * 0.01)))
ax.yaxis.set_major_formatter('${x:1.2f}')
ax.tick_params(axis='x', which='minor', bottom=False)
ax.set_facecolor('#faf0e6')
fig.patch.set_facecolor('#faf0e6')
if c.current_hour <= 14:
    plt.xticks(np.arange(days + 1), graphing_dates[:days + 1], rotation=40)
else:
    plt.xticks(np.arange(days + 1), graphing_dates[1:days + 2], rotation=40)
x_ticks = ax.xaxis.get_major_ticks()
# x_ticks[0].label.set_visible(False)
plt.title(f"{symbol} Prediction", fontweight='bold')
plt.xlabel("Dates", fontweight='bold')
plt.ylabel("Closing Price", fontweight='bold')
plt.show()
plt.savefig('prediction.jpg')
