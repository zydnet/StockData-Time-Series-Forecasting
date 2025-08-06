"""
Main training script for Amazon Stock Data Time Series Forecasting
Demonstrates efficient model training and evaluation using optimized components
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import our optimized modules
from config import *
from utils import *
from data_pipeline import DataPipeline
from models import ModelFactory
from evaluation import ModelEvaluator

class StockPricePredictor:
    """Main class for stock price prediction using optimized components"""
    
    def __init__(self):
        self.config = {
            'DATA_CONFIG': DATA_CONFIG,
            'LSTM_CONFIG': LSTM_CONFIG,
            'GRU_CONFIG': GRU_CONFIG,
            'ARIMA_CONFIG': ARIMA_CONFIG,
            'TRAINING_CONFIG': TRAINING_CONFIG,
            'METRICS_CONFIG': METRICS_CONFIG
        }
        
        self.data_pipeline = DataPipeline(self.config)
        self.model_factory = ModelFactory(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data using optimized pipeline"""
        print("Loading and preparing data...")
        
        # Load data
        df = self.data_pipeline.load_data(DATA_CONFIG['amzn_file'])
        print(f"Loaded {len(df)} data points")
        
        # Engineer features
        df = self.data_pipeline.engineer_features(df)
        print(f"Engineered features: {df.shape[1]} columns")
        
        return df
    
    def train_lstm_model(self, X_train, y_train, X_val, y_val):
        """Train LSTM model with optimized configuration"""
        print("Training LSTM model...")
        
        # Create model
        model = self.model_factory.create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        
        # Get callbacks
        callbacks = self.model_factory.get_callbacks()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=LSTM_CONFIG['epochs'],
            batch_size=LSTM_CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_gru_model(self, X_train, y_train, X_val, y_val):
        """Train GRU model with optimized configuration"""
        print("Training GRU model...")
        
        # Create model
        model = self.model_factory.create_gru_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        
        # Get callbacks
        callbacks = self.model_factory.get_callbacks()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=GRU_CONFIG['epochs'],
            batch_size=GRU_CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_arima_model(self, train_data):
        """Train ARIMA model"""
        print("Training ARIMA model...")
        
        from models import ARIMAModel
        
        model = ARIMAModel(
            order=ARIMA_CONFIG['order'],
            seasonal_order=ARIMA_CONFIG.get('seasonal_order')
        )
        
        model.fit(train_data)
        return model
    
    def evaluate_models(self, models_dict, X_test, y_test, test_data=None):
        """Evaluate all trained models"""
        print("Evaluating models...")
        
        for model_name, model_info in models_dict.items():
            if 'lstm' in model_name.lower() or 'gru' in model_name.lower():
                # Neural network models
                y_pred = model_info['model'].predict(X_test)
                y_pred = self.data_pipeline.inverse_transform_predictions(y_pred)
                y_true = self.data_pipeline.inverse_transform_predictions(y_test)
                
            elif 'arima' in model_name.lower():
                # ARIMA model
                steps = len(test_data)
                y_pred = model_info['model'].predict(steps)
                y_true = test_data.values[-steps:]
                
            else:
                # Other models
                y_pred = model_info['model'].predict(X_test)
                y_true = y_test
            
            # Evaluate model
            metrics = self.evaluator.evaluate_model(model_name, y_true, y_pred)
            self.results[model_name] = metrics
            
            # Plot predictions
            self.evaluator.plot_predictions(
                y_true, y_pred, 
                title=f"{model_name} Predictions"
            )
            
            # Plot residuals
            self.evaluator.plot_residuals(
                y_true, y_pred,
                title=f"{model_name} Residuals Analysis"
            )
            
            print(f"{model_name} - RMSE: {metrics['RMSE']:.4f}, "
                  f"Directional Accuracy: {metrics['Directional_Accuracy']:.4f}")
    
    def run_complete_pipeline(self):
        """Run complete training and evaluation pipeline"""
        print("Starting complete stock price prediction pipeline...")
        print("=" * 60)
        
        # 1. Load and prepare data
        df = self.load_and_prepare_data()
        
        # 2. Prepare sequences
        X, y, scalers = self.data_pipeline.prepare_sequences(
            df, 
            target_col='Close',
            lookback=LSTM_CONFIG['lookback_days']
        )
        
        # 3. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_pipeline.split_data(
            X, y,
            test_size=TRAINING_CONFIG['test_size'],
            validation_size=0.1
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # 4. Train models
        models_dict = {}
        
        # Train LSTM
        lstm_model, lstm_history = self.train_lstm_model(X_train, y_train, X_val, y_val)
        models_dict['LSTM'] = {'model': lstm_model, 'history': lstm_history}
        
        # Train GRU
        gru_model, gru_history = self.train_gru_model(X_train, y_train, X_val, y_val)
        models_dict['GRU'] = {'model': gru_model, 'history': gru_history}
        
        # Train ARIMA (on original scale data)
        train_data = df['Close'][:len(X_train) + LSTM_CONFIG['lookback_days']]
        arima_model = self.train_arima_model(train_data)
        models_dict['ARIMA'] = {'model': arima_model}
        
        # 5. Evaluate models
        test_data = df['Close'][len(X_train) + LSTM_CONFIG['lookback_days']:]
        self.evaluate_models(models_dict, X_test, y_test, test_data)
        
        # 6. Generate comparison report
        self.evaluator.generate_report(self.results)
        
        # 7. Plot model comparison
        self.evaluator.plot_model_comparison(
            self.results, 
            metric='RMSE',
            title="Model Comparison - RMSE"
        )
        
        # 8. Save results
        self.save_results(models_dict)
        
        print("Pipeline completed successfully!")
        return models_dict, self.results
    
    def save_results(self, models_dict):
        """Save models and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for model_name, model_info in models_dict.items():
            if 'lstm' in model_name.lower() or 'gru' in model_name.lower():
                model_path = MODELS_DIR / f"{model_name.lower()}_{timestamp}.h5"
                model_info['model'].save(model_path)
                print(f"Saved {model_name} to {model_path}")
        
        # Save results
        results_df = pd.DataFrame(self.results).T
        results_path = RESULTS_DIR / f"results_{timestamp}.csv"
        results_df.to_csv(results_path)
        print(f"Saved results to {results_path}")
        
        # Save evaluation report
        report_path = RESULTS_DIR / f"evaluation_report_{timestamp}.txt"
        self.evaluator.generate_report(self.results, str(report_path))
    
    def create_ensemble_model(self, models_dict, X_test, y_test):
        """Create and evaluate ensemble model"""
        print("Creating ensemble model...")
        
        # Get predictions from all models
        predictions = {}
        for model_name, model_info in models_dict.items():
            if 'lstm' in model_name.lower() or 'gru' in model_name.lower():
                pred = model_info['model'].predict(X_test)
                pred = self.data_pipeline.inverse_transform_predictions(pred)
                predictions[model_name] = pred
        
        # Create ensemble
        ensemble_model = self.model_factory.create_ensemble_model(
            list(models_dict.values()),
            weights=[0.4, 0.3, 0.3]  # Adjust weights based on performance
        )
        
        # Evaluate ensemble
        y_true = self.data_pipeline.inverse_transform_predictions(y_test)
        ensemble_pred = ensemble_model.predict(X_test)
        ensemble_pred = self.data_pipeline.inverse_transform_predictions(ensemble_pred)
        
        metrics = self.evaluator.evaluate_model('Ensemble', y_true, ensemble_pred)
        self.results['Ensemble'] = metrics
        
        return ensemble_model

def main():
    """Main function to run the complete pipeline"""
    predictor = StockPricePredictor()
    
    try:
        models_dict, results = predictor.run_complete_pipeline()
        
        # Create ensemble model
        ensemble_model = predictor.create_ensemble_model(
            models_dict, 
            predictor.X_test, 
            predictor.y_test
        )
        
        print("\nFinal Results:")
        print("=" * 40)
        for model_name, metrics in predictor.results.items():
            print(f"{model_name}:")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.4f}")
            print()
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 