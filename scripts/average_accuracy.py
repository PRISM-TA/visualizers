### This is to calculate the average accuracy across all stocks for one model and feature set
### To run this script, execute: python scripts/average_accuracy.py --model 'model' --features 'feature_set'

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sqlalchemy import select, create_engine, distinct
from models.ClassifierResult import ClassifierResult
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import warnings
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import numpy as np
from sqlalchemy.orm import sessionmaker
import argparse

# Load environment variables from .env
load_dotenv()

# Construct database URL
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

warnings.filterwarnings('ignore', category=UserWarning)

def calculate_metrics(y_true, y_pred) -> Dict:
    """Calculate metrics with proper handling of undefined cases"""
    metrics = {}
    
    # Calculate metrics with zero_division parameter
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calculate per-class metrics
    for label in range(3):  # 0: downtrend, 1: sideways, 2: uptrend
        class_mask = y_true == label
        if np.any(class_mask):
            metrics[f'recall_class_{label}'] = recall_score(y_true == label, y_pred == label, zero_division=0)
            metrics[f'precision_class_{label}'] = precision_score(y_true == label, y_pred == label, zero_division=0)
            metrics[f'f1_class_{label}'] = f1_score(y_true == label, y_pred == label, zero_division=0)
    
    return metrics

def get_acronym(feature_set: str) -> str:
    """Create acronym from feature set name."""
    # Split by spaces and '+' and get first letter of each word
    words = feature_set.replace('+', ' ').split()
    acronym = ''.join(word[0].upper() for word in words)
    return acronym

def get_all_tickers(db_session, model_name: str, feature_set: str) -> List[str]:
    """Get all available tickers for the given model and feature set."""
    with db_session() as session:
        query = (
            select(distinct(ClassifierResult.ticker))
            .where(
                ClassifierResult.model == model_name,
                ClassifierResult.feature_set == feature_set
            )
        )
        
        result = session.execute(query).all()
        return [r[0] for r in result]

def get_stock_results(db_session, ticker: str, model_name: str, feature_set: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get actual and predicted labels for a specific stock."""
    with db_session() as session:
        query = (
            select(ClassifierResult.actual_label, ClassifierResult.predicted_label)
            .where(
                ClassifierResult.ticker == ticker,
                ClassifierResult.model == model_name,
                ClassifierResult.feature_set == feature_set
            )
            .order_by(ClassifierResult.report_date)
        )
        
        query_result = session.execute(query).all()
        
        if not query_result:
            return np.array([]), np.array([])

        # Extract actual and predicted labels
        y_true = np.array([record[0] for record in query_result])
        y_pred = np.array([record[1] for record in query_result])
        
        return y_true, y_pred

def analyze_average_accuracy(db_session, model_name: str, feature_set: str) -> None:
    """Analyze and print average classifier results across all stocks."""
    try:
        # Create output directory if it doesn't exist
        output_dir = "average_accuracy_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename using acronym
        feature_acronym = get_acronym(feature_set)
        filename = f"{output_dir}/average_{model_name}_{feature_acronym}.txt"
        
        # Redirect stdout to both file and console
        class Logger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "w", encoding='utf-8')
            
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                
            def flush(self):
                self.terminal.flush()
                self.log.flush()
        
        sys.stdout = Logger(filename)
        
        # Get all available tickers for this model and feature set
        tickers = get_all_tickers(db_session, model_name, feature_set)
        
        if not tickers:
            print(f"No results found for model '{model_name}' with feature set '{feature_set}'")
            return
        
        print("\n" + "="*50)
        print(f"Average Accuracy Analysis")
        print(f"Model: {model_name}")
        print(f"Feature Set: {feature_set}")
        print("="*50)
        
        print(f"\nNumber of stocks analyzed: {len(tickers)}")
        print(f"Stocks: {', '.join(tickers)}")
        
        # Collect all predictions and actual values
        all_y_true = []
        all_y_pred = []
        
        # Create per-stock metrics table
        stock_metrics = []
        
        # Process each stock
        for ticker in tickers:
            y_true, y_pred = get_stock_results(db_session, ticker, model_name, feature_set)
            
            if len(y_true) == 0:
                continue
                
            # Add to aggregate arrays
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            # Calculate individual stock metrics
            accuracy = np.mean(y_true == y_pred)
            metrics = calculate_metrics(y_true, y_pred)
            
            stock_metrics.append({
                'ticker': ticker,
                'predictions': len(y_true),
                'accuracy': accuracy,
                'recall': metrics['recall'],
                'precision': metrics['precision'],
                'f1': metrics['f1']
            })
        
        # Convert to arrays for overall metrics
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        if len(all_y_true) == 0:
            print("No valid prediction data found.")
            return
            
        # Calculate overall metrics
        overall_accuracy = np.mean(all_y_true == all_y_pred)
        overall_metrics = calculate_metrics(all_y_true, all_y_pred)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2])
        
        # Print overall results
        print(f"\nTotal Predictions Across All Stocks: {len(all_y_true)}")
        print(f"Overall Average Accuracy: {overall_accuracy:.4f}")
        
        # Calculate and print class distribution
        print("\nActual Label Distribution:")
        unique, counts = np.unique(all_y_true, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Label {label}: {count} ({count/len(all_y_true)*100:.2f}%)")
            
        print("\nPredicted Label Distribution:")
        unique, counts = np.unique(all_y_pred, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Label {label}: {count} ({count/len(all_y_pred)*100:.2f}%)")
        
        # Print overall metrics
        print("\nOverall Metrics:")
        print(f"Macro Recall: {overall_metrics['recall']:.4f}")
        print(f"Macro Precision: {overall_metrics['precision']:.4f}")
        print(f"Macro F1: {overall_metrics['f1']:.4f}")
        
        # Print per-class metrics
        class_names = ['Downtrend', 'Sideways', 'Uptrend']
        for i, name in enumerate(class_names):
            if f'recall_class_{i}' in overall_metrics:
                print(f"\n{name}:")
                print(f"  Recall: {overall_metrics[f'recall_class_{i}']:.4f}")
                print(f"  Precision: {overall_metrics[f'precision_class_{i}']:.4f}")
                print(f"  F1: {overall_metrics[f'f1_class_{i}']:.4f}")
        
        # Print overall confusion matrix
        print("\nOverall Confusion Matrix:")
        conf_df = pd.DataFrame(
            conf_matrix,
            index=[f'True_{c}' for c in class_names],
            columns=[f'Pred_{c}' for c in class_names]
        )
        print(conf_df)
        
        # Print per-stock metrics table
        print("\nPer-Stock Metrics:")
        metrics_df = pd.DataFrame(stock_metrics)
        if not metrics_df.empty:
            metrics_df = metrics_df.sort_values('accuracy', ascending=False)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', 120)
            print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            
            # Calculate and print summary statistics
            print("\nAccuracy Statistics:")
            print(f"Mean Accuracy: {metrics_df['accuracy'].mean():.4f}")
            print(f"Median Accuracy: {metrics_df['accuracy'].median():.4f}")
            print(f"Min Accuracy: {metrics_df['accuracy'].min():.4f} ({metrics_df.loc[metrics_df['accuracy'].idxmin(), 'ticker']})")
            print(f"Max Accuracy: {metrics_df['accuracy'].max():.4f} ({metrics_df.loc[metrics_df['accuracy'].idxmax(), 'ticker']})")
            print(f"Standard Deviation: {metrics_df['accuracy'].std():.4f}")

        # Reset stdout
        sys.stdout = sys.stdout.terminal

        print(f"\nResults saved to: {filename}")

    except Exception as e:
        print(f"Error analyzing results: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze average accuracy across all stocks')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--features', required=True, help='Feature set name')
    
    args = parser.parse_args()
    
    engine = create_engine(DB_URL)
    DBSession = sessionmaker(bind=engine)
    
    analyze_average_accuracy(
        db_session=DBSession,
        model_name=args.model,
        feature_set=args.features
    )