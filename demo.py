"""
Demo script for Amazon Stock Data Time Series Forecasting
Simple demonstration of the optimized system
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our optimized modules
from config import *
from utils import *
from data_pipeline import DataPipeline
from models import ModelFactory
from evaluation import ModelEvaluator

def run_demo():
    """Run a simple demo of the optimized system"""
    print("🚀 Amazon Stock Data Time Series Forecasting - Demo")
    print("=" * 60)
    
    try:
        # 1. Initialize components
        print("1. Initializing components...")
        config = {
            'DATA_CONFIG': DATA_CONFIG,
            'LSTM_CONFIG': LSTM_CONFIG,
            'TRAINING_CONFIG': TRAINING_CONFIG
        }
        
        data_pipeline = DataPipeline(config)
        model_factory = ModelFactory(config)
        evaluator = ModelEvaluator(config)
        
        # 2. Load and prepare data
        print("2. Loading and preparing data...")
        df = data_pipeline.load_data('AMZN.csv')
        print(f"   Loaded {len(df)} data points")
        
        # Show basic data info
        print(f"   Data shape: {df.shape}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
        
        # 3. Engineer features
        print("3. Engineering features...")
        df = data_pipeline.engineer_features(df)
        print(f"   Created {df.shape[1]} features")
        
        # Show some technical indicators
        tech_indicators = ['ma_7', 'ma_21', 'macd', 'rsi', 'bb_upper', 'bb_lower']
        available_indicators = [col for col in tech_indicators if col in df.columns]
        print(f"   Technical indicators: {available_indicators}")
        
        # 4. Prepare sequences (small sample for demo)
        print("4. Preparing sequences...")
        # Use only last 1000 data points for demo
        df_sample = df.tail(1000)
        X, y, scalers = data_pipeline.prepare_sequences(
            df_sample, 
            target_col='Close',
            lookback=30  # Smaller lookback for demo
        )
        
        print(f"   Created {len(X)} sequences")
        print(f"   Input shape: {X.shape}")
        
        # 5. Split data
        print("5. Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline.split_data(
            X, y,
            test_size=0.2,
            validation_size=0.1
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Validation set: {X_val.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # 6. Create and train a simple model
        print("6. Creating and training model...")
        model = model_factory.create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            config={'units': [32, 16, 1], 'epochs': 10}  # Smaller model for demo
        )
        
        # Train with fewer epochs for demo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            verbose=1
        )
        
        # 7. Make predictions
        print("7. Making predictions...")
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions
        y_pred_original = data_pipeline.inverse_transform_predictions(y_pred)
        y_test_original = data_pipeline.inverse_transform_predictions(y_test)
        
        # 8. Evaluate model
        print("8. Evaluating model...")
        metrics = evaluator.calculate_regression_metrics(y_test_original, y_pred_original)
        
        print("\n📊 Model Performance:")
        print(f"   RMSE: ${metrics['RMSE']:.2f}")
        print(f"   MAE: ${metrics['MAE']:.2f}")
        print(f"   Directional Accuracy: {metrics['Directional_Accuracy']:.2%}")
        print(f"   Hit Ratio: {metrics['Hit_Ratio']:.2f}%")
        
        # 9. Create visualizations
        print("9. Creating visualizations...")
        
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_original, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred_original, label='Predicted', color='red', alpha=0.7)
        plt.title('Amazon Stock Price Prediction - Demo')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('demo_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot learning curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('demo_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 10. Save results
        print("10. Saving results...")
        results_df = pd.DataFrame([metrics])
        results_df.to_csv('demo_results.csv', index=False)
        print("   Results saved to 'demo_results.csv'")
        
        print("\n✅ Demo completed successfully!")
        print("📁 Generated files:")
        print("   - demo_predictions.png")
        print("   - demo_learning_curves.png")
        print("   - demo_results.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_data_info():
    """Show information about the available data"""
    print("\n📊 Data Information:")
    print("=" * 40)
    
    try:
        # Load data
        df = pd.read_csv('AMZN.csv')
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Number of trading days: {len(df)}")
        
        # Show price statistics
        print(f"\nPrice Statistics:")
        print(f"  Min Close: ${df['Close'].min():.2f}")
        print(f"  Max Close: ${df['Close'].max():.2f}")
        print(f"  Mean Close: ${df['Close'].mean():.2f}")
        print(f"  Std Close: ${df['Close'].std():.2f}")
        
        # Show recent data
        print(f"\nRecent data (last 5 rows):")
        print(df.tail()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_string(index=False))
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    print("🎯 Amazon Stock Data Time Series Forecasting - Demo")
    print("=" * 60)
    
    # Show data information
    show_data_info()
    
    # Run demo
    print("\n" + "=" * 60)
    success = run_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("You can now run the full training pipeline with:")
        print("python main_training.py")
    else:
        print("\n❌ Demo failed. Please check the error messages above.") 