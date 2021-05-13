import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import math
from keras import Sequential
from keras.layers import LSTM, Dense
dataframe = pd.read_csv("DataFrame.csv")
dataset = dataframe.reset_index()["close"]
# plt.plot(dataset)
"""
plt.figure(figsize=(16, 8))
plt.title("History")
plt.plot(dataframe["close"])
plt.xlabel("Date")
plt.ylabel("Close Price ")
plt.show()
"""
data = dataframe.filter(["close"])
dataset = data.values

training_data_len = math.ceil(len(dataset) * .8)
training_data_len


Normalization = MinMaxScaler(feature_range=(0, 1))

scaled_data = Normalization.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]
print(scaled_data)
# Splitting the data
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()
print(len(x_train))

# To numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping the x_train
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=False))

model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=50, epochs=80)

# Testing data
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = Normalization.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse

# Plotting the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid["Predictions"] = predictions
# Visualizing
plt.figure(figsize=(16, 8))
plt.title("Model")
plt.xlabel("Date")
plt.ylabel("CLOSE PRICE")
plt.plot(train["close"])
plt.plot(valid[['close', 'Predictions']])

