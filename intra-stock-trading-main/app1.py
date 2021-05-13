import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import keras.models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv


def ts_train_test_normalize(all_data, time_steps):
    # create training and test set

    ts_train = all_data
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


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST', "GET"])
def stock_analysis():
    model = keras.models.load_model(r"C:\Users\Um Ar\PycharmProjects\Internship-2\Project 2\rnn.pkl")
    if request.method == "POST":
        data = request.files['file']
        input_data = pd.read_csv(data, parse_dates=["Date"])
        df = ts_train_test_normalize(input_data, time_steps=5)
        output = model.predict(df)
        output = pd.DataFrame(output)
        output = output.to_html()
        # plt.plot(output)
        # plt.savefig("output.png")
        return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)
