import keras.models
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
data = yf.download(tickers="MSFT")
import matplotlib.pyplot as plt
print(data)
data.to_csv("MSFT.csv")

data = pd.read_csv("MSFT.csv", parse_dates=["Date"])


def ts_train_test_normalize(all_data, time_steps):
    # create training and test set

    ts_train = all_data
    ts_train = ts_train.drop("High", axis=1)
    ts_train = ts_train.drop("Adj Close", axis=1)
    ts_train = ts_train.drop("Close", axis=1)
    ts_train = ts_train.drop("Low", axis=1)
    ts_train = ts_train.drop("Volume", axis=1)
    ts_train = all_data.iloc[:, 0:1].values

    ts_train_len = len(ts_train)
    # scale the data
    sc = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train_stacked = []
    for i in range(time_steps, ts_train_len - 1):
        X_train.append(ts_train_scaled[i - time_steps:i, 0])
    X_train = np.array(X_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train


Data = ts_train_test_normalize(data, time_steps=5)
print(Data.shape)

model = keras.models.load_model(r"C:\Users\Um Ar\PycharmProjects\Internship-2\Project 2\rnn.pkl")
predict = model.predict(Data)
print(predict)
plt.plot(predict)
plt.plot(data["Adj Close"])
