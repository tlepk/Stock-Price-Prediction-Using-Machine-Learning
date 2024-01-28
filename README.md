Stock Price Prediction Using Machine Learning
Introduction
This project employs logistic regression to predict stock movements, aiming to enhance investment decision-making. By analyzing historical data and key features, we explore the algorithm's effectiveness in capturing stock price fluctuations. The study aims to provide insights into leveraging machine learning for more informed trading strategies in dynamic financial markets.
Objective
Prediction of Thai stock movements ERW, TISCO, SPRC, time window every 15 minutes.
Use Logistic Regression Model to make the decision Binary classification only incase Buy (1) and Sell (-1) of Close price in each 15 minutes.
Code Overview
Stock Price Prediction Using Machine Learning 1

 Import The Libraries :
Imports libraries for data manipulation, technical indicators, standardization, plotting, and machine learning. We will start by importing the necessary libraries such as TA-Lib. It sets up tools for working with financial data, including technical indicators using the 'ta' library and fetching data from Yahoo Finance with 'y finance'.
  # Data Manipulation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Technical Indicators
import ta
#Standardize
from sklearn.preprocessing import StandardScaler
Stock Price Prediction Using Machine Learning 2

 # Plotting graphs
import matplotlib.pyplot as plt
import plotly.express as px
# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
# Data fetching
import yfinance as yf
yf.pdr_override()
import pendulum
Import The Data
This code fetches historical stock data for the company with the ticker specified by stock from Yahoo Finance within the given date range (from December 11, 2023, to January 11, 2024) with a 15-minute interval.
 # List of stock tickers for different companies
company = ['erw.bk', 'tisco.bk', 'sprc.bk']
# Selecting a specific stock by index from the list
# (0 for The Erawan Group, 1 for TISCO Financial Group,
# 2 for Star Petroleum Refining Public)
stock = company[0]  # Change index to select a different compan
# Accessing stock data for the selected company from Yahoo Fina
stk = yf.Ticker(stock)
# Defining the start and end dates for retrieving historical da
start = pendulum.parse('2023-12-11 10:00')
end = pendulum.parse('2024-01-11 16:30')
Stock Price Prediction Using Machine Learning
3
y n
t

 # Retrieving historical stock data with a 15-minute
# interval for the specified date range
df = stk.history(interval='15m', start=start, end=end)
# Selecting the first 5 columns in the dataframe
df = df.iloc[:, :5]
# Displaying information about the dataframe,
# including data types and non-null counts
df.info()
Define Feature
This function takes a DataFrame (data) with financial data, calculates Exponential Moving Average (EMA), Relative Strength Index (RSI), Bollinger Bands (Upper, Middle, Lower), Stochastic Oscillator, Average True Range (ATR), On-Balance Volume (OBV), and Rate of Change (ROC) based on specified windows, and returns an updated DataFrame with these indicators added as columns.
o i
o 0 o w o
a l
 def indicators(data) :
    # Exponential Moving Average (EMA) with a window of 20 peri
    data['EMA'] = ta.trend.ema_indicator(close=data['Close'], w
    # Relative Strength Index (RSI) with a window of 20 periods
        data['RSI'] = ta.momentum.rsi(close=data['Close'], wind
    # Bollinger Bands (Upper, Middle, Lower) with a window of 2
        data['BB_Upper'], _, data['BB_Lower'] = ta.volatility.b
        ta.volatility.bollinger_mavg(close=data['Close'], windo
        ta.volatility.bollinger_lband(close=data['Close'], wind
    # Stochastic Oscillator with a window of 20 periods
        data['Stoch_Oscillator'] = ta.momentum.stoch(close=data
        high=data['High'], low=data['Low'], window=20)
    # Average True Range (ATR) with a window of 20 periods
        data['ATR'] = ta.volatility.average_true_range(high=dat
        low=data['Low'], close=data['Close'], window=20)
    # On-Balance Volume (OBV)
        data['OBV'] = ta.volume.on_balance_volume(close=data['C
Stock Price Prediction Using Machine Learning
4

     volume=data['Volume'])
# Rate of Change (ROC) with a window of 20 periods
  data['ROC'] = ta.momentum.roc(close=data['Close'], window
# Drop rows with NaN values after adding indicators
    data = data.dropna()
return data
Let us print the top five rows of column
Feature
1. Exponential Moving Average (EMA): Gives more weight to recent prices,
making it more responsive to price changes compared to SMA. Feature: EMA
2. Relative Strength Index (RSI): Indicates whether a stock is overbought or oversold, helping to identify potential reversal points.
Feature: RSI
3. Bollinger Bands (BB): Consists of three lines (upper, middle, lower)
representing volatility and potential price reversal points. Features: BB_Middle , BB_Upper , BB_Lower
4. Stochastic Oscillator: Compares a closing price to its price range over a specified period, indicating momentum and potential reversal points.
Feature: Stoch_Oscillator
5. Average True Range (ATR): Measures market volatility by considering the
range between high, low, and previous close prices over a specified period. Feature: ATR
6. On-Balance Volume (OBV): Uses volume flow to predict changes in stock price, indicating buying or selling pressure.
=
     Stock Price Prediction Using Machine Learning
5

Feature: OBV
7. Price Rate of Change (ROC): Measures the percentage change in price between the current price and the price a certain number of periods ago, indicating momentum.
Feature: ROC Define Target
This code snippet creates a binary target variable 'Y' based on the comparison of the next day's closing prices with the current day's closing prices. If the next day's closing price is greater, it assigns 1 (indicating a "buy" signal); otherwise, it assigns -1 (indicating a "sell" signal). The resulting binary values are then stored in a DataFrame named 'y'.
Standardize Data
This code snippet demonstrates the standardization of features using a StandardScaler . The standardized values are then stored in a new DataFrame,
'x_scaled', which can be used for further analysis or as input to machine learning models.
e
 # Creating a binary target variable 'Y' based on the comparison
Y = np.where(X['Close'].shift(-1) > X['Close'],1,-1)
# Create 'y' with a column named 'buy' based on the binary targ
y = pd.DataFrame({'buy(1) or sell(-1)': Y})
  # Creating a StandardScaler object
scaler = StandardScaler()
# Scaling (standardizing) the features in the DataFrame 'X'
x_scaled = scaler.fit_transform(X)
# Creating 'x_scaled' with the scaled features and using the or
x_scaled = pd.DataFrame(x_scaled, columns=X.columns)
Split Data
Stock Price Prediction Using Machine Learning
6
i

To calculates the index where the data will be split, allocating 80% for training and 20% for testing. This code snippet is responsible for splitting the data into training and testing sets. The split is stratified, ensuring that the class distribution is maintained, and a random seed is set for reproducibility.
Instantiate The Logistic Regression in Python
This code creates, trains, and examines the coefficients of a logistic regression model. The comments provide an overview of each step, including the creation of the logistic regression model, training it on the training data, retrieving the learned coefficients, and organizing the results into a Pandas Data Frame
e e
 X_train, X_test, y_train, y_test = train_test_split(x_scaled,
                                                    y, test_siz
                                                    stratify=y,
                                                    random_stat
 # Creating a Logistic Regression model
model = LogisticRegression()
# Fitting the model to the training data
model = model.fit (X_train,y_train)
#Examine The Coefficients
pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
Calculate Class Probabilities & Predict Class Labels
The code demonstrates using the trained logistic regression model to obtain both predicted probabilities ( predict_proba ) and final predicted class labels ( predict ) for a test dataset.
   # Predict probabilities for each class using the test data
probability = model.predict_proba(X_test)
print(probability)
# Predict the class labels for the test data
Stock Price Prediction Using Machine Learning
7

 predicted = model.predict(X_test)
print(predicted)
Evaluate The Model
Confusion Matrix: metrics.confusion_matrix(y_test, predicted) and Classification Report: metrics.classification_report(y_test, predicted) . Both outputs are crucial for evaluating the effectiveness of a classification model. The confusion matrix gives a detailed breakdown of correct and incorrect predictions, while the classification report provides a summary of key metrics for each class and overall performance.
   # Print the confusion matrix to evaluate the classifier's perfo
print(metrics.confusion_matrix(y_test, predicted))
# Print a classification report showing various metrics
print(metrics.classification_report(y_test, predicted))
Model Accuracy
This code snippet prints the accuracy of the trained logistic regression model on the test dataset.
AUC
This code calculates the Receiver Operating Characteristic (ROC) curve for a binary classification model and plots it using Plotly.
r
 # Print the accuracy of the trained model on the test dataset
print(model.score(X_test, y_test))
 from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
predicted_probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, predicted_probabilitie
# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)
Stock Price Prediction Using Machine Learning
8
s

 roc_auc = auc(fpr, tpr)
print(f'Area Under the Curve (AUC) is {roc_auc}')
# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
Stock Price Prediction Using Machine Learning
9
a
