"""
Comprehensive evaluation framework for Amazon Stock Data Time Series Forecasting
Provides multiple metrics, visualizations, and model comparison tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def calculate_regression_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
        
        Returns:
            dict: Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(y_true) > 0
        predicted_direction = np.diff(y_pred) > 0
        directional_accuracy = accuracy_score(actual_direction, predicted_direction)
        
        # Hit ratio (percentage of correct directional predictions)
        hit_ratio = np.sum(actual_direction == predicted_direction) / len(actual_direction) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Hit_Ratio': hit_ratio
        }
    
    def calculate_classification_metrics(self, y_true, y_pred):
        """
        Calculate classification metrics for directional prediction
        
        Args:
            y_true (np.array): True directions (1 for up, 0 for down)
            y_pred (np.array): Predicted directions
        
        Returns:
            dict: Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
    
    def evaluate_model(self, model_name, y_true, y_pred, model_type='regression'):
        """
        Evaluate a single model
        
        Args:
            model_name (str): Name of the model
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            model_type (str): Type of model ('regression' or 'classification')
        
        Returns:
            dict: Evaluation results
        """
        if model_type == 'regression':
            metrics = self.calculate_regression_metrics(y_true, y_pred)
        else:
            metrics = self.calculate_classification_metrics(y_true, y_pred)
        
        self.results[model_name] = metrics
        return metrics
    
    def compare_models(self, results_dict):
        """
        Compare multiple models
        
        Args:
            results_dict (dict): Dictionary of model results
        
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_df = pd.DataFrame(results_dict).T
        
        # Add ranking columns
        for metric in comparison_df.columns:
            if metric in ['MSE', 'RMSE', 'MAE', 'MAPE']:
                comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank()
            else:
                comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df
    
    def plot_predictions(self, y_true, y_pred, title="Model Predictions", 
                        save_path=None):
        """
        Plot actual vs predicted values
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            title (str): Plot title
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, title="Residuals Analysis", 
                      save_path=None):
        """
        Plot residual analysis
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            title (str): Plot title
            save_path (str): Path to save plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Residuals over time
        axes[1, 1].plot(residuals)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals over Time')
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_dict, metric='RMSE', 
                             title="Model Comparison", save_path=None):
        """
        Plot model comparison bar chart
        
        Args:
            results_dict (dict): Dictionary of model results
            metric (str): Metric to compare
            title (str): Plot title
            save_path (str): Path to save plot
        """
        models = list(results_dict.keys())
        values = [results_dict[model][metric] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values)
        
        # Color bars based on performance
        colors = ['green' if v == min(values) else 'red' if v == max(values) else 'blue' 
                 for v in values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title(title)
        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, history, title="Learning Curves", save_path=None):
        """
        Plot learning curves for neural network models
        
        Args:
            history: Training history from Keras model
            title (str): Plot title
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results_dict, save_path=None):
        """
        Generate comprehensive evaluation report
        
        Args:
            results_dict (dict): Dictionary of model results
            save_path (str): Path to save report
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model comparison table
        comparison_df = self.compare_models(results_dict)
        report.append("MODEL COMPARISON:")
        report.append("-" * 40)
        report.append(comparison_df.to_string())
        report.append("")
        
        # Best model for each metric
        report.append("BEST MODELS BY METRIC:")
        report.append("-" * 40)
        for metric in ['RMSE', 'MAE', 'R2', 'Directional_Accuracy']:
            if metric in comparison_df.columns:
                best_model = comparison_df[metric].idxmin() if metric in ['RMSE', 'MAE'] else comparison_df[metric].idxmax()
                best_value = comparison_df.loc[best_model, metric]
                report.append(f"{metric}: {best_model} ({best_value:.4f})")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text
    
    def calculate_confidence_intervals(self, y_true, y_pred, confidence=0.95):
        """
        Calculate confidence intervals for predictions
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            confidence (float): Confidence level
        
        Returns:
            dict: Confidence interval statistics
        """
        residuals = y_true - y_pred
        std_residuals = np.std(residuals)
        
        # Calculate confidence intervals
        z_score = 1.96  # For 95% confidence
        margin_of_error = z_score * std_residuals
        
        return {
            'std_residuals': std_residuals,
            'margin_of_error': margin_of_error,
            'confidence_interval': (y_pred - margin_of_error, y_pred + margin_of_error)
        } 