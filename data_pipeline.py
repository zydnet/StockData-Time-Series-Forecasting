"""
Optimized data pipeline for Amazon Stock Data Time Series Forecasting
Handles data loading, preprocessing, feature engineering, and sequence creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

class DataPipeline:
    """Optimized data pipeline for time series forecasting"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.feature_scaler = None
        
    def load_data(self, file_path, date_column='Date'):
        """
        Load and prepare data with optimized memory usage
        
        Args:
            file_path (str): Path to CSV file
            date_column (str): Name of date column
        
        Returns:
            pd.DataFrame: Prepared dataframe
        """
        # Optimize data types for memory efficiency
        dtype_dict = {
            'Open': 'float32',
            'High': 'float32', 
            'Low': 'float32',
            'Close': 'float32',
            'Volume': 'int32'
        }
        
        def parser(x):
            return pd.to_datetime(x, format='%m/%d/%Y')
        
        df = pd.read_csv(
            file_path, 
            parse_dates=[date_column],
            date_parser=parser,
            dtype=dtype_dict
        )
        
        # Set index and sort
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def engineer_features(self, df):
        """
        Create comprehensive feature set with optimized calculations
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Lag features
        df = self._add_lag_features(df)
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        return df
    
    def _add_technical_indicators(self, df):
        """Add technical indicators efficiently"""
        # Moving averages
        for window in [7, 14, 21, 30]:
            df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Exponential moving averages
        for span in [12, 26]:
            df[f'ema_{span}'] = df['Close'].ewm(span=span).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_price_features(self, df):
        """Add price-based features"""
        # Price ratios
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Price changes
        df['price_change'] = df['Close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Price momentum
        for period in [1, 3, 5, 10]:
            df[f'momentum_{period}'] = df['Close'].pct_change(periods=period)
        
        return df
    
    def _add_volume_features(self, df):
        """Add volume-based features"""
        # Volume moving averages
        for window in [5, 10, 20]:
            df[f'volume_ma_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
        
        # Volume price trend
        df['vpt'] = (df['Volume'] * df['price_change']).cumsum()
        
        return df
    
    def _add_volatility_features(self, df):
        """Add volatility features"""
        # Rolling volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window=window).std()
        
        # True Range
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift())
        df['tr3'] = abs(df['Low'] - df['Close'].shift())
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Remove temporary columns
        df.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
        
        return df
    
    def _add_lag_features(self, df):
        """Add lag features"""
        # Price lags
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        return df
    
    def prepare_sequences(self, df, target_col='Close', lookback=60, 
                         feature_cols=None, target_scaling=True):
        """
        Prepare sequences for deep learning models
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            lookback (int): Number of time steps to look back
            feature_cols (list): Feature columns to use
            target_scaling (bool): Whether to scale target variable
        
        Returns:
            tuple: (X, y, scalers)
        """
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        # Remove any columns that don't exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Prepare features and target
        features = df[feature_cols].values
        target = df[target_col].values.reshape(-1, 1)
        
        # Remove any rows with NaN values
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target).any(axis=1))
        features = features[valid_mask]
        target = target[valid_mask]
        
        if len(features) == 0:
            raise ValueError("No valid data after removing NaN values")
        
        # Scale features
        self.feature_scaler = MinMaxScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Scale target if requested
        if target_scaling:
            self.scaler = MinMaxScaler()
            target_scaled = self.scaler.fit_transform(target)
        else:
            target_scaled = target
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(features_scaled)):
            X.append(features_scaled[i-lookback:i])
            y.append(target_scaled[i, 0])
        
        return np.array(X), np.array(y), (self.feature_scaler, self.scaler)
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.1):
        """
        Split data into train, validation, and test sets
        
        Args:
            X (np.array): Feature sequences
            y (np.array): Target values
            test_size (float): Proportion of test set
            validation_size (float): Proportion of validation set
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        total_size = len(X)
        test_split = int(total_size * (1 - test_size))
        val_split = int(test_split * (1 - validation_size))
        
        X_train, X_val, X_test = X[:val_split], X[val_split:test_split], X[test_split:]
        y_train, y_val, y_test = y[:val_split], y[val_split:test_split], y[test_split:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_time_series_cv(self, n_splits=5):
        """
        Create time series cross-validation splits
        
        Args:
            n_splits (int): Number of splits
        
        Returns:
            TimeSeriesSplit: Cross-validation object
        """
        return TimeSeriesSplit(n_splits=n_splits)
    
    def inverse_transform_predictions(self, predictions, scaler=None):
        """
        Inverse transform predictions to original scale
        
        Args:
            predictions (np.array): Scaled predictions
            scaler: Scaler object used for scaling
        
        Returns:
            np.array: Predictions in original scale
        """
        if scaler is None:
            scaler = self.scaler
        
        if scaler is not None:
            return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        else:
            return predictions
    
    def get_feature_importance(self, model, feature_names):
        """
        Get feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (list): List of feature names
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            return importance_df.sort_values('importance', ascending=False)
        else:
            return None 