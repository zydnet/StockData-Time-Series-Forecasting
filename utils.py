"""
Utility functions for Amazon Stock Data Time Series Forecasting
Centralized module for common functions used across notebooks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import datetime
import warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path, date_column='Date', target_column='Close'):
    """
    Load and prepare stock data with proper date formatting
    
    Args:
        file_path (str): Path to CSV file
        date_column (str): Name of date column
        target_column (str): Name of target column (usually Close)
    
    Returns:
        pd.DataFrame: Prepared dataframe with datetime index
    """
    def parser(x):
        return datetime.datetime.strptime(x, '%m/%d/%Y')
    
    df = pd.read_csv(file_path, header=0, parse_dates=[0], date_parser=parser)
    df.set_index(date_column, inplace=True)
    return df

def get_technical_indicators(dataset):
    """
    Generate technical indicators for stock data
    
    Args:
        dataset (pd.DataFrame): Stock data with OHLCV columns
    
    Returns:
        pd.DataFrame: Dataset with technical indicators added
    """
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['Close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])
    
    # Create Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(window=20).std()
    dataset['upper_band'] = (dataset['Close'].rolling(window=20).mean()) + (dataset['20sd'] * 2)
    dataset['lower_band'] = (dataset['Close'].rolling(window=20).mean()) - (dataset['20sd'] * 2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = (dataset['Close'] / 100) - 1
    
    return dataset

def create_sequences(data, target_col, lookback=60):
    """
    Create sequences for time series prediction
    
    Args:
        data (pd.DataFrame): Input data
        target_col (str): Target column name
        lookback (int): Number of time steps to look back
    
    Returns:
        tuple: (X, y) sequences
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[target_col].iloc[i])
    return np.array(X), np.array(y)

def normalize_data(data, scaler=None):
    """
    Normalize data using MinMaxScaler
    
    Args:
        data (np.array): Input data
        scaler (MinMaxScaler, optional): Pre-fitted scaler
    
    Returns:
        tuple: (normalized_data, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()
        return scaler.fit_transform(data), scaler
    else:
        return scaler.transform(data), scaler

def plot_predictions(actual, predicted, title="Stock Price Prediction"):
    """
    Plot actual vs predicted values
    
    Args:
        actual (np.array): Actual values
        predicted (np.array): Predicted values
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def calculate_metrics(actual, predicted):
    """
    Calculate various performance metrics
    
    Args:
        actual (np.array): Actual values
        predicted (np.array): Predicted values
    
    Returns:
        dict: Dictionary containing various metrics
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    # Calculate directional accuracy
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    directional_accuracy = accuracy_score(actual_direction, predicted_direction)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Directional_Accuracy': directional_accuracy
    }

def plot_technical_indicators(dataset, last_days=400):
    """
    Plot technical indicators
    
    Args:
        dataset (pd.DataFrame): Dataset with technical indicators
        last_days (int): Number of last days to plot
    """
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0 - last_days
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)
    
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['Close'], label='Closing Price', color='b')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title(f'Technical indicators for Amazon - last {last_days} days.')
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['momentum'], label='Momentum', color='b', linestyle='-')
    plt.legend()
    plt.show()

def fourier_transform_analysis(data, close_col='Close'):
    """
    Perform Fourier transform analysis on time series data
    
    Args:
        data (pd.DataFrame): Input data
        close_col (str): Name of close price column
    
    Returns:
        tuple: (fft_df, reconstructed_signals)
    """
    close_fft = np.fft.fft(np.asarray(data[close_col].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    
    # Reconstruct signals with different components
    fft_list = np.asarray(fft_df['fft'].tolist())
    reconstructed_signals = {}
    
    for num_ in [3, 6, 9, 100]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_: -num_] = 0
        reconstructed_signals[f'{num_}_components'] = np.fft.ifft(fft_list_m10)
    
    return fft_df, reconstructed_signals 