"""
Model factory for Amazon Stock Data Time Series Forecasting
Centralized model architectures with configurable parameters
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

class ModelFactory:
    """Factory class for creating different types of models"""
    
    def __init__(self, config):
        self.config = config
        
    def create_lstm_model(self, input_shape, config=None):
        """
        Create LSTM model
        
        Args:
            input_shape (tuple): Input shape (timesteps, features)
            config (dict): Model configuration
        
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        if config is None:
            config = self.config.get('LSTM_CONFIG', {})
            
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=config.get('units', [50])[0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(config.get('dropout_rate', 0.2)))
        
        # Additional LSTM layers
        for i, units in enumerate(config.get('units', [50, 50])[1:-1]):
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(config.get('dropout_rate', 0.2)))
        
        # Final LSTM layer
        model.add(LSTM(config.get('units', [50, 50, 1])[-2]))
        model.add(Dropout(config.get('dropout_rate', 0.2)))
        
        # Output layer
        model.add(Dense(config.get('units', [50, 50, 1])[-1]))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_gru_model(self, input_shape, config=None):
        """
        Create GRU model
        
        Args:
            input_shape (tuple): Input shape (timesteps, features)
            config (dict): Model configuration
        
        Returns:
            tf.keras.Model: Compiled GRU model
        """
        if config is None:
            config = self.config.get('GRU_CONFIG', {})
            
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(
            units=config.get('units', [50])[0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(config.get('dropout_rate', 0.2)))
        
        # Additional GRU layers
        for i, units in enumerate(config.get('units', [50, 50])[1:-1]):
            model.add(GRU(units, return_sequences=True))
            model.add(Dropout(config.get('dropout_rate', 0.2)))
        
        # Final GRU layer
        model.add(GRU(config.get('units', [50, 50, 1])[-2]))
        model.add(Dropout(config.get('dropout_rate', 0.2)))
        
        # Output layer
        model.add(Dense(config.get('units', [50, 50, 1])[-1]))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_bidirectional_lstm(self, input_shape, config=None):
        """
        Create Bidirectional LSTM model
        
        Args:
            input_shape (tuple): Input shape (timesteps, features)
            config (dict): Model configuration
        
        Returns:
            tf.keras.Model: Compiled Bidirectional LSTM model
        """
        if config is None:
            config = self.config.get('LSTM_CONFIG', {})
            
        model = Sequential()
        
        # Bidirectional LSTM layers
        model.add(Bidirectional(
            LSTM(config.get('units', [50])[0], return_sequences=True),
            input_shape=input_shape
        ))
        model.add(Dropout(config.get('dropout_rate', 0.2)))
        
        model.add(Bidirectional(
            LSTM(config.get('units', [50, 50])[1], return_sequences=False)
        ))
        model.add(Dropout(config.get('dropout_rate', 0.2)))
        
        # Output layer
        model.add(Dense(config.get('units', [50, 50, 1])[-1]))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_ensemble_model(self, models, weights=None):
        """
        Create ensemble model
        
        Args:
            models (list): List of trained models
            weights (list): Weights for each model
        
        Returns:
            EnsembleModel: Ensemble model
        """
        if weights is None:
            weights = [1/len(models)] * len(models)
            
        return EnsembleModel(models, weights)
    
    def get_callbacks(self, config=None):
        """
        Get training callbacks
        
        Args:
            config (dict): Configuration dictionary
        
        Returns:
            list: List of callbacks
        """
        if config is None:
            config = self.config.get('LSTM_CONFIG', {})
            
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.get('early_stopping_patience', 10),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        return callbacks

class EnsembleModel:
    """Ensemble model that combines multiple models"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
    def predict(self, X):
        """
        Make ensemble prediction
        
        Args:
            X: Input data
        
        Returns:
            np.array: Ensemble predictions
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
            
        return ensemble_pred

class ARIMAModel:
    """ARIMA model wrapper"""
    
    def __init__(self, order=(5, 1, 0), seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        
    def fit(self, data):
        """
        Fit ARIMA model
        
        Args:
            data (pd.Series): Time series data
        """
        if self.seasonal_order:
            self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
        else:
            self.model = ARIMA(data, order=self.order)
        
        self.fitted_model = self.model.fit()
        
    def predict(self, steps):
        """
        Make predictions
        
        Args:
            steps (int): Number of steps to predict
        
        Returns:
            np.array: Predictions
        """
        return self.fitted_model.forecast(steps=steps)

class RandomForestModel:
    """Random Forest model for time series"""
    
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        
    def fit(self, X, y):
        """Fit the model"""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

class LinearRegressionModel:
    """Linear Regression model for time series"""
    
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
        
    def fit(self, X, y):
        """Fit the model"""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X) 