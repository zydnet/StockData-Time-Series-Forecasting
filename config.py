"""
Configuration file for Amazon Stock Data Time Series Forecasting
Centralized configuration for all hyperparameters and settings
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    try:
        dir_path.mkdir(exist_ok=True)
    except FileExistsError:
        pass  # Directory already exists

# Data configuration
DATA_CONFIG = {
    'amzn_file': 'AMZN.csv',
    'close_file': 'Close.csv',
    'date_column': 'Date',
    'target_column': 'Close',
    'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'technical_indicators': ['ma7', 'ma21', 'MACD', 'upper_band', 'lower_band', 'ema', 'momentum']
}

# Model hyperparameters
LSTM_CONFIG = {
    'lookback_days': 60,
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'learning_rate': 0.001,
    'dropout_rate': 0.2,
    'units': [50, 50, 1],  # LSTM units for each layer
    'activation': 'relu',
    'optimizer': 'adam'
}

GRU_CONFIG = {
    'lookback_days': 60,
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'learning_rate': 0.001,
    'dropout_rate': 0.2,
    'units': [50, 50, 1],  # GRU units for each layer
    'activation': 'tanh',
    'optimizer': 'adam'
}

ARIMA_CONFIG = {
    'order': (5, 1, 0),  # (p, d, q)
    'seasonal_order': (1, 1, 1, 12),  # (P, D, Q, s)
    'trend': 'c'  # 'n', 'c', 't', 'ct'
}

# Training configuration
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'shuffle': False,  # Important for time series
    'scaler': 'minmax',  # 'minmax', 'standard', 'robust'
    'cross_validation_folds': 5
}

# Evaluation metrics
METRICS_CONFIG = {
    'regression_metrics': ['mse', 'rmse', 'mae', 'mape'],
    'classification_metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'directional_accuracy': True
}

# Visualization configuration
PLOT_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis'
}

# API configuration (if using external data sources)
API_CONFIG = {
    'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY', ''),
    'yahoo_finance': True,
    'update_frequency': 'daily'
}

# Sentiment analysis configuration
SENTIMENT_CONFIG = {
    'max_sequence_length': 100,
    'embedding_dim': 100,
    'vocab_size': 10000,
    'lstm_units': 128,
    'dropout_rate': 0.3,
    'batch_size': 32,
    'epochs': 10
}

# GAN configuration (experimental)
GAN_CONFIG = {
    'latent_dim': 100,
    'generator_layers': [256, 512, 1024],
    'discriminator_layers': [512, 256, 128],
    'batch_size': 32,
    'epochs': 1000,
    'critic_iterations': 5,
    'lambda_gp': 10
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'technical_indicators': True,
    'fourier_transform': True,
    'rolling_windows': [7, 14, 21, 30],
    'lag_features': [1, 2, 3, 5, 10],
    'price_ratios': True,
    'volatility_features': True
}

# Model comparison configuration
COMPARISON_CONFIG = {
    'models': ['lstm', 'gru', 'arima', 'fourier', 'ensemble'],
    'baseline_model': 'naive_forecast',
    'ensemble_method': 'weighted_average',
    'weights': [0.4, 0.3, 0.2, 0.1]  # For ensemble models
} 