import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import keras.models
from sklearn.preprocessing import MinMaxScaler


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


colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]


@app.route('/predict', methods=['POST', "GET"])
def stock_analysis():
    model = keras.models.load_model(r"GRU_NFTY_1min.pkl")
    if request.method == "POST":
        data = request.files['file']
        input_data = pd.read_csv(data)
        df = ts_train_test_normalize(input_data, time_steps=5)
        output = model.predict(df)
        return render_template('index.html',
                               prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)
