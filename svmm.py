import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import pickle
import yfinance
data = yfinance.download("MSFT")
data.head()
data.tail()
data["Adj Close"].plot()
plt.show()
print(len(data))
train = pd.read_csv("1mindata/MSFT1Min.csv")
test = pd.read_csv("1mindata/1MSFT-Test.csv")
len(train)
len(test)

train = train.sort_values('Datetime')
test = test.sort_values("Datetime")

train.reset_index(inplace=True)
train.set_index("Datetime", inplace=True)

test.reset_index(inplace=True)
test.set_index("Datetime", inplace=True)
"""
plt.figure(figsize=(12, 6))
plt.plot(train["Adj Close"])
plt.xlabel('Date', fontsize=15)
plt.ylabel('Adjusted Close Price', fontsize=15)
plt.show()


# Rolling mean
close_px = train['Adj Close']
mavg = close_px.rolling(window=100).mean()

plt.figure(figsize=(12, 6))
close_px.plot(label='MSFT')
mavg.plot(label='mavg')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
"""

dates_df = train.copy()
dates_df = dates_df.reset_index()
org_dates = dates_df['Datetime']

# convert to ints
dates_df['Datetime'] = dates_df['Datetime'].map(mdates.date2num)

dates_df.tail()

dates = dates_df['Datetime'].to_numpy()
prices = train['Adj Close'].to_numpy()

test_df = test.copy()
test_df = test_df.reset_index()
org_dates_test = test_df['Datetime']
test_df['Datetime'] = test_df['Datetime'].map(mdates.date2num)
test_dates = test_df['Datetime'].to_numpy()
test_prices = test['Adj Close'].to_numpy()

test_dates = np.reshape(test_dates, (len(test_dates), 1))
test_prices = np.reshape(test_prices, (len(test_prices), 1))


# Convert to 1d Vector
dates = np.reshape(dates, (len(dates), 1))
prices = np.reshape(prices, (len(prices), 1))

clf = SVR(kernel='linear')
clf.fit(dates, prices)


plt.figure(figsize=(12, 6))
plt.plot(dates, prices, color='black', label='Data')
plt.plot(org_dates, clf.predict(dates), color='red', label='Linear model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
predictions = clf.predict(test_dates)
with open('svm_linear.pickle', 'wb') as f:
    pickle.dump(clf, f)
print(mean_squared_error(test_prices, predictions))
