import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import SGD
df = pd.read_csv("DataFrame.csv")
df.head()

print(df['Date'].dtype)

data = df.filter(['close'])

# Convert dataframe to numpy
dataset = data.values

training_data_len = math.ceil(len(dataset) * 0.75)
print(training_data_len)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

time_step=1
# create scaled training dataset
train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []

for i in range(time_step, len(train_data) ):
  x_train.append(train_data[i-time_step:i, 0])
  y_train.append(train_data[i, 0])

# converting training data to numpy for using LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Creating the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer=SGD(momentum=0.9), loss='mean_squared_error')
# Training the model
model.fit(x_train, y_train, batch_size=10, epochs=100, validation_split=0.14)
test_data = scaled_data[training_data_len - time_step:, :]

x_test = []
y_test = dataset[training_data_len:, :]


for i in range(time_step, len(test_data)):
  x_test.append(test_data[i-time_step:i, 0])

# convert to numpy and reshape
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Error
rmse = np.sqrt((np.mean(predictions - y_test)**2))
print("The error rate of the model is", rmse)
