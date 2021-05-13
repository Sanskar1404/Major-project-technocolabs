import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import yfinance as yf
import matplotlib.dates as mdates
import pickle


def preprocessing():
    data = yf.download("MSFT", period="1d", interval='1m' )
    print(data.columns)
    data.sort_values('Datetime')
    data.reset_index(inplace=True)
    data.set_index("Datetime", inplace=True)
    dates_df = data.copy()
    dates_df = dates_df.reset_index()
    dates_df['Datetime'] = dates_df['Datetime'].map(mdates.date2num)
    dates = dates_df['Datetime'].to_numpy()
    dates = np.reshape(dates, (len(dates), 1))
    return dates


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST', "GET"])
def stock_analysis():
    model = pickle.load(open("svm_linear.pickle", 'rb'))
    if request.method == "POST":
        data = preprocessing()
        result = model.predict(data)
        return render_template('output.html', prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
