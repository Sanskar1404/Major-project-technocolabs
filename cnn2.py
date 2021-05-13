import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from keras import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix,\
    classification_report, max_error, mean_absolute_error, mean_squared_error
from keras.layers import Dense, RNN, GRU, Dropout, Conv1D, Conv2D, Conv3D, MaxPooling2D, MaxPooling1D
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import datetime
import pickle
df = pd.read_csv("DataFrame.csv")
df = df.drop("Type", axis=1)
df = df.drop("classification", axis=1)
df["Time"] = pd.to_datetime(df['Time'])

df1 = df.reset_index()['close']
print(df1.describe)


scalar = MinMaxScaler(feature_range=(0, 1))
df1 = scalar.fit_transform(np.array(df1).reshape(-1, 1))
training_size = int(len(df1)*0.85)
test_size = len(df1)-training_size
train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

print(training_size, test_size)


def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(Conv3D(124, return_sequences=True, input_shape=(100, 1)))
model.add(Conv3D(64, return_sequences=True))
model.add(Conv3D(124))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scalar.inverse_transform(train_predict)
test_predict = scalar.inverse_transform(test_predict)
math.sqrt(mean_squared_error(y_train, train_predict))
math.sqrt(mean_squared_error(ytest, test_predict))

# model.save("lstm.h5")
# Plotting
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scalar.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


def plot_predictions(test, predicted):
    plt.plot(test, color="red", label="Reak")
    plt.plot(predicted, color="green", label="predicted")
    plt.title("stock price prediction")
    plt.xlabel("time")
    plt.ylabel("stock price")
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("the root mean squared error is : {}.".format(rmse))


plot_predictions(ytest, test_predict)

return_rmse(ytest, test_predict)
