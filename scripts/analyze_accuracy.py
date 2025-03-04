### This is to calculate the accuracy matrix for each stock/model/feature_set based on the uploaded classifier results
### To run this script, execute: python scripts/analyze_accuracy.py --ticker 'ticker' --model 'model' --features 'feature_set'

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sqlalchemy import select, create_engine
from models.ClassifierResult import ClassifierResult
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import warnings
from typing import Dict
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

def analyze_results(db_session, ticker: str, model_name: str, feature_set: str) -> None:
    """Analyze and print classifier results."""
    try:
        # Create output directory if it doesn't exist
        output_dir = "accuracy_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename using acronym
        feature_acronym = get_acronym(feature_set)
        filename = f"{output_dir}/{ticker}_{model_name}_{feature_acronym}.txt"
        
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
        
        with db_session as session:
            # Query using join pattern
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
                print(f"No results found for {ticker} - {model_name} - {feature_set}")
                return

            # Extract actual and predicted labels
            y_true = np.array([record[0] for record in query_result])
            y_pred = np.array([record[1] for record in query_result])

            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            accuracy = np.mean(y_true == y_pred)
            
            # Print results
            print("\n" + "="*50)
            print(f"Results Analysis for {ticker}")
            print(f"Model: {model_name}")
            print(f"Feature Set: {feature_set}")
            print("="*50)
            
            print(f"\nTotal Predictions: {len(query_result)}")
            print(f"Overall Accuracy: {accuracy:.4f}")
            
            # Calculate and print class distribution
            print("\nActual Label Distribution:")
            unique, counts = np.unique(y_true, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"Label {label}: {count} ({count/len(y_true)*100:.2f}%)")
                
            print("\nPredicted Label Distribution:")
            unique, counts = np.unique(y_pred, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"Label {label}: {count} ({count/len(y_pred)*100:.2f}%)")
            
            # Calculate metrics
            metrics = calculate_metrics(y_true, y_pred)
            
            print("\nMetrics:")
            print(f"Macro Recall: {metrics['recall']:.4f}")
            print(f"Macro Precision: {metrics['precision']:.4f}")
            print(f"Macro F1: {metrics['f1']:.4f}")
            
            # Print per-class metrics
            class_names = ['Downtrend', 'Sideways', 'Uptrend']
            for i, name in enumerate(class_names):
                if f'recall_class_{i}' in metrics:
                    print(f"\n{name}:")
                    print(f"  Recall: {metrics[f'recall_class_{i}']:.4f}")
                    print(f"  Precision: {metrics[f'precision_class_{i}']:.4f}")
                    print(f"  F1: {metrics[f'f1_class_{i}']:.4f}")
            
            # Print confusion matrix
            print("\nConfusion Matrix:")
            class_names = ['Downtrend', 'Sideways', 'Uptrend']
            conf_df = pd.DataFrame(
                conf_matrix,
                index=[f'True_{c}' for c in class_names],
                columns=[f'Pred_{c}' for c in class_names]
            )
            print(conf_df)

        # Reset stdout
        sys.stdout = sys.stdout.terminal

        print(f"\nResults saved to: {filename}")

    except Exception as e:
        print(f"Error analyzing results: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze classifier results')
    parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--features', required=True, help='Feature set name')
    
    args = parser.parse_args()
    
    engine = create_engine(DB_URL)
    DBSession = sessionmaker(bind=engine)
    
    analyze_results(
        db_session=DBSession,
        ticker=args.ticker,
        model_name=args.model,
        feature_set=args.features
    )
