import keras.models
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = yf.download(tickers="MSFT")
# data = pd.read_csv("DataFrame.csv")
# Print the data
print(data.describe())
print(data.head())
print(data.tail())
print(data.columns)

print(data["Adj Close"])


def ts_train_test(all_data, time_steps, for_periods):
    # create training and test set
    ts_train = all_data[:'2017'].iloc[:, 0:1].values
    ts_test = all_data['2018':].iloc[:, 0:1].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps, ts_train_len - 1):
        X_train.append(ts_train[i - time_steps:i, 0])
        y_train.append(ts_train[i:i + for_periods, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Preparing to create X_test
    inputs = pd.concat((all_data["Adj Close"][:'2017'],
                        all_data["Adj Close"]['2018':]), axis=0).values
    inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1, 1)

    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i - time_steps:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test


X_train, y_train, X_test = ts_train_test(data, 5, 2)
print(X_train.shape[0], X_train.shape[1])
print(len(X_train))
print(len(X_test))

# Convert the 3-D shape of X_train to a data frame so we can see:
X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0], X_train.shape[1])))
y_train_see = pd.DataFrame(y_train)
pd.concat([X_train_see, y_train_see], axis=1)

# Convert the 3-D shape of X_test to a data frame so we can see:
X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0],X_test.shape[1])))
pd.DataFrame(X_test_see)

print("There are " + str(X_train.shape[0]) + " samples in the training data")
print("There are " + str(X_test.shape[0]) + " samples in the test data")


def ts_train_test_normalize(all_data, time_steps, for_periods):
    # create training and test set
    ts_train = all_data[:'2017'].iloc[:, 0:1].values
    ts_test = all_data['2018':].iloc[:, 0:1].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # scale the data
    sc = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps, ts_train_len - 1):
        X_train.append(ts_train_scaled[i - time_steps:i, 0])
        y_train.append(ts_train_scaled[i:i + for_periods, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    inputs = pd.concat((all_data["Adj Close"][:'2017'],
                        all_data["Adj Close"]['2018':]), axis=0).values
    inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i - time_steps:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, sc


def simple_rnn_model(X_train, y_train, X_test):

    my_rnn_model = Sequential()
    my_rnn_model.add(LSTM(64, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
    my_rnn_model.add(LSTM(64, activation="tanh", return_sequences=True))
    my_rnn_model.add(LSTM(64, activation="tanh", return_sequences=True))
    my_rnn_model.add(LSTM(64, activation="tanh", return_sequences=True))
    my_rnn_model.add(LSTM(32))
    my_rnn_model.add(Dense(1))

    my_rnn_model.compile(optimizer="adam", loss='mean_squared_error')

    # fit the RNN model
    my_rnn_model.fit(X_train, y_train, epochs=100, batch_size=150, verbose=0)

    # Finalizing predictions
    rnn_predictions = my_rnn_model.predict(X_test)

    return my_rnn_model, rnn_predictions


my_rnn_model, rnn_predictions = simple_rnn_model(X_train, y_train, X_test)
rnn_predictions[1:10]


def actual_pred_plot(preds):

    actual_pred = pd.DataFrame(columns=['Adj. Close', 'prediction'])
    actual_pred['Adj. Close'] = data.loc['2018':, 'Adj Close'][0:len(preds)]
    actual_pred['prediction'] = preds[:, 0]

    from keras.metrics import MeanSquaredError
    m = MeanSquaredError()
    m.update_state(np.array(actual_pred['Adj. Close']), np.array(actual_pred['prediction']))

    return (m.result().numpy(), actual_pred.plot())


actual_pred_plot(rnn_predictions)


def simple_rnn_model(X_train, y_train, X_test, sc):

    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN

    my_rnn_model = Sequential()
    my_rnn_model.add(LSTM(32, return_sequences=True))
    # my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    # my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    my_rnn_model.add(SimpleRNN(32))
    my_rnn_model.add(Dense(2))  # The time step of the output

    my_rnn_model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
    my_rnn_model.fit(X_train, y_train, epochs=10, batch_size=150, verbose=1)

    # Finalizing predictions
    rnn_predictions = my_rnn_model.predict(X_test)
    from sklearn.preprocessing import MinMaxScaler
    rnn_predictions = sc.inverse_transform(rnn_predictions)

    return my_rnn_model, rnn_predictions


X_train, y_train, X_test, sc = ts_train_test_normalize(data, 5, 2)
my_rnn_model, rnn_predictions_2 = simple_rnn_model(X_train, y_train, X_test, sc)
print(rnn_predictions_2[1:10])
actual_pred_plot(rnn_predictions_2)

my_rnn_model.save("rnn.pkl")

model = keras.models.load_model(r"C:\Users\Um Ar\PycharmProjects\Internship-2\Project 2\rnn.pkl")
Xtest, ytest = train_test_split(data, test_size=0.2)
predictions = model.predict(X_train)
print(X_train.shape)
rmse = np.sqrt((np.mean(predictions - y_train)**2))
print("The error rate of the model is", rmse)
