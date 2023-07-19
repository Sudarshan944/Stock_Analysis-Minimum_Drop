# Import necessary libraries
from math import *
from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta

# Initialize Flask application
app = Flask(__name__)

# Load NLTK resources for sentiment analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Define routes


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    stock_symbol = request.form['stock_symbol']
    news_text = request.form['news_text']

    # Perform sentiment analysis on news text
    sentiment_score = sia.polarity_scores(news_text)['compound']
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    # Perform stock price prediction (replace this with your own prediction model)
    predicted_price = predict_stock_price(stock_symbol)

    # Render the result template with predictions
    return render_template('result.html', stock_symbol=stock_symbol, news_text=news_text,
                           sentiment=sentiment, predicted_price=predicted_price)


def moving_average(data):
    window = 20
    n = len(data)
    moving_avg = []
    if n < window:
        return None  # Return None if there is insufficient data for the moving average calculation

    for i in range(window, n):
        avg = sum(data[i - window + 1: i + 1]) / window
        moving_avg.append(round(avg, 2))

    return moving_avg


def predict_stock_price(stock_symbol):
    # Define the number of days to look back for calculating the moving average
    lookback_days = 1000

    # Fetch historical stock data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    data = yf.download(stock_symbol, start=start_date,
                       end=end_date, progress=False)

    # Check if data is available
    if len(data) == 0:
        return "No historical data available for the specified stock symbol."
    print(len(data))
    # Extract the closing prices from the historical data
    closing_prices = data['Close'].tolist()

    # Calculate the moving average
    moving_avg = moving_average(closing_prices)

    # Check if moving_avg is empty or contains insufficient data
    if moving_avg is None or len(moving_avg) == 0:
        return "Insufficient data to predict the stock price."

    # Predict the next day's stock price as the moving average of the last available day
    predicted_price = moving_avg[-1]

    return predicted_price




# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
