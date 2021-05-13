import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from keras import Sequential
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import math
from keras import Sequential
from keras.layers import LSTM, Dense, RNN

dataframe = pd.read_csv("DataFrame.csv")
dataframe["change"] = dataframe[["open", "close"]].pct_change(axis=1)["close"]
clas = []


for col in dataframe["change"]:
    if col <= 0:
        clas.append("-1")
    elif col > 0:
        clas.append("1")
    else:
        clas.append("NA")

dataframe["classification"] = clas

print(dataframe.classification)
dataframe.to_csv("Dataframe.csv")

plt.plot(dataframe)

Normalizer = MinMaxScaler(feature_range=(0, 1))
dataframe = Normalizer.fit_transform(np.array(dataframe)).reshape(-1, 1)

# Training and testing
training_size = int(len(dataframe)*0.65)
test_size = len(dataframe)-training_size
train_data, test_data = dataframe[0:training_size, :], dataframe[training_size:len(dataframe), :1]


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i:(i + time_step), 0])
    return np.array(dataX), numpy.array(dataY)


time_step = 1
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# REshaping
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_train.shape)
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=False))

model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=50, epochs=80)
