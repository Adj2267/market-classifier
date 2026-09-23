import sys
import argparse
import yfinance as yf
import pandas as pd
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_lnl(X, y, theta):
    h = sigmoid(X @ theta)
    lnl = np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    return lnl

def compute_gradient(X, y, theta):
    h = sigmoid(X @ theta)
    gradient = X.T @ (y - h)
    return gradient

def compute_hessian(X, theta):
    h = sigmoid(X @ theta)
    hessian = -X.T @ (h * (1-h) * X)
    return hessian

def main():
    # Set up argument parser, Take cmdline argument for stock ticker
    parser = argparse.ArgumentParser("Provide a stock ticker for analysis")
    parser.add_argument("ticker", help="Stock ticker symbol")

    args = parser.parse_args()

    # Download historical stock data using yfinance
    try:
        data = yf.download(args.ticker, period="max")
        if data.empty:
            print(f"No data found for {args.ticker}")
            sys.exit(1)
    except Exception as e:
        print(f"Error occurred while downloading data for {args.ticker}: {e}")
        sys.exit(1)

    # Calculate momentum and volatility features
    data['Mom_5'] = data['Close'].pct_change(periods=5)
    data['Mom_20'] = data['Close'].pct_change(periods=20)
    data['Vol_20'] = data['Close'].pct_change(periods=1).rolling(window=20).std()

    # Create target variable based on 5-day momentum
    data['Target'] = np.where(data['Mom_5'].shift(-5) > 0, 1, 0)
    data.dropna(inplace=True)

    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    X_train = data[['Mom_5', 'Mom_20', 'Vol_20']].iloc[:train_size]
    y_train = data['Target'].iloc[:train_size]

    X_test = data[['Mom_5', 'Mom_20', 'Vol_20']].iloc[train_size:]
    y_test = data['Target'].iloc[train_size:]

    #normalize the features
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    theta = np.zeros((X_train.shape[1], 1))
    m,n = X_train.shape

    for i in range(10):
        gradient = compute_gradient(X_train, y_train, theta)
        hessian = compute_hessian(X_train, theta)
        update = np.linalg.solve(hessian, gradient)
        theta -= update
        print(f"Iteration {i+1}: Log-Likelihood = {compute_lnl(X_train, y_train, theta)}")

    h_test = sigmoid(X_test @ theta)
    h_test = np.where(h_test >= 0.5, 1, 0)
    accuracy = np.mean(h_test == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    main()