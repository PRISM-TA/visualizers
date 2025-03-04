### This is to visualize the stock price graph with the real and predicted labels

import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import warnings
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
from sqlalchemy import select
from ..models.SupervisedClassifierDataset import SupClassifierDataset
from ..models.MarketData import MarketData

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def calculate_metrics(y_true, y_pred):
    """Calculate metrics with proper handling of undefined cases"""
    metrics = {}

    # Calculate metrics with zero_division parameter
    metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return metrics


def create_trend_visualization(
    db_session,
    ticker: str,
    results: List[Dict],
    save_dir: str = "visualizations",
    time_window: Tuple[str, str] = None,
    overall_accuracy: float = None,
) -> None:
    """
    Create visualization comparing real trends, model predictions, and stock prices.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get real labels from database
    stmt = select(SupClassifierDataset).where(SupClassifierDataset.ticker == ticker)
    result = db_session.execute(stmt)

    real_labels = []
    for row in result.scalars():
        real_labels.append(
            {"start_date": row.start_date, "end_date": row.end_date, "label": row.label}
        )

    real_labels_df = pd.DataFrame(real_labels)
    real_labels_df["start_date"] = pd.to_datetime(real_labels_df["start_date"])
    real_labels_df["end_date"] = pd.to_datetime(real_labels_df["end_date"])

    # Create predictions DataFrame
    pred_df = pd.DataFrame(results)
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # Get the date range from predictions
    pred_start_date = pred_df["date"].min()
    pred_end_date = pred_df["date"].max()

    # Get market data for the same period
    market_stmt = (
        select(MarketData)
        .where(
            MarketData.ticker == ticker,
            MarketData.report_date >= pred_start_date,
            MarketData.report_date <= pred_end_date,
        )
        .order_by(MarketData.report_date)
    )

    market_result = db_session.execute(market_stmt)
    market_data = []
    for row in market_result.scalars():
        market_data.append(
            {"date": row.report_date, "close": row.close, "volume": row.volume}
        )

    market_df = pd.DataFrame(market_data)
    market_df["date"] = pd.to_datetime(market_df["date"])

    # Filter real labels to match prediction date range
    real_labels_df = real_labels_df[
        (real_labels_df["start_date"] >= pred_start_date)
        & (real_labels_df["start_date"] <= pred_end_date)
    ]

    # Create visualization with three subplots
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

    # Plot 1: Stock Price
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(market_df["date"], market_df["close"], "b-", label="Stock Price")
    ax0.set_title(f"{ticker} Stock Price")
    ax0.set_ylabel("Price")
    ax0.grid(True)
    ax0.legend()

    # Add volume as bar chart on secondary y-axis
    ax0_volume = ax0.twinx()
    ax0_volume.bar(
        market_df["date"], market_df["volume"], alpha=0.3, color="gray", label="Volume"
    )
    ax0_volume.set_ylabel("Volume")

    # Plot 2: Real Trends
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    colors = ["red", "yellow", "green"]
    labels = ["Downtrend", "Sideways", "Uptrend"]

    for label in range(3):
        mask = real_labels_df["label"] == label
        ax1.scatter(
            real_labels_df[mask]["start_date"],
            real_labels_df[mask]["label"],
            c=colors[label],
            alpha=0.5,
            label=labels[label],
        )

    ax1.set_title("Real Market Trends")
    ax1.set_ylabel("Trend Label")
    ax1.legend()
    ax1.grid(True)

    # Plot 3: Predictions
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    for label in range(3):
        mask = pred_df["prediction"] == label
        ax2.scatter(
            pred_df[mask]["date"],
            pred_df[mask]["prediction"],
            c=colors[label],
            alpha=0.5,
            label=labels[label],
        )

    ax2.set_title("Predicted Market Trends")
    ax2.set_ylabel("Predicted Label")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True)

    # Format x-axis
    plt.xticks(rotation=45)

    # Add confidence visualization on secondary y-axis for predictions
    ax2_conf = ax2.twinx()
    ax2_conf.plot(
        pred_df["date"], pred_df["confidence"], "b--", alpha=0.5, label="Confidence"
    )
    ax2_conf.set_ylabel("Prediction Confidence")
    ax2_conf.legend(loc="center right")

    # Adjust layout and save
    plt.tight_layout()
    # Get start date and format it as YYYYMMDD
    start_date = pred_df["date"].min().strftime("%Y%m%d")
    max_predictions = len(results)

    # Format accuracy as percentage with 2 decimal places
    accuracy_str = (
        f"{overall_accuracy*100:.2f}" if overall_accuracy is not None else "NA"
    )

    # Create filename with start_date and max_predictions
    filename = f"trend_comparison_{start_date}_{max_predictions}_{accuracy_str}.png"
    save_path = os.path.join(save_dir, filename)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print metrics to console
    print(f"\nVisualization saved to: {save_path}")
    print(f"Start date: {start_date}")
    print(f"Number of predictions: {max_predictions}")
    print(f"\nMetrics:")


def calculate_comparison_stats(real_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict:
    """Calculate comparison statistics between real and predicted labels."""
    return {
        "total_predictions": len(pred_df),
        "prediction_distribution": pred_df["prediction"].value_counts().to_dict(),
    }


def save_statistics(stats: Dict, save_dir: str, timestamp: str) -> None:
    """Save statistics to a text file."""
    stats_path = os.path.join(save_dir, f"statistics_{timestamp}.txt")
    with open(stats_path, "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
