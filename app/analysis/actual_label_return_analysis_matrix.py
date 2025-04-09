# actual_labels_return_plot.py

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from typing import Dict, List
import sys
from dotenv import load_dotenv
import kaleido
from plotly.subplots import make_subplots

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary models
from models.ClassifierResult import ClassifierResult
from models.MarketData import MarketData
from models.EquityIndicators import EquityIndicators

# Load environment variables
load_dotenv()

# Construct database URL 
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

def get_classifier_results(db_session, ticker: str, model: str, feature_set: str) -> pd.DataFrame:
    """Get all classifier results for a specific ticker, model, and feature set."""
    # Query classifier results
    query = select(
        ClassifierResult.report_date,
        ClassifierResult.predicted_label,
        ClassifierResult.actual_label,
        ClassifierResult.uptrend_prob,
        ClassifierResult.side_prob,
        ClassifierResult.downtrend_prob
    ).where(
        (ClassifierResult.ticker == ticker) &
        (ClassifierResult.model == model) &
        (ClassifierResult.feature_set == feature_set)
    ).order_by(ClassifierResult.report_date)
    
    # Execute query and convert to DataFrame
    classifier_results = pd.DataFrame(db_session.execute(query).all())
    
    if classifier_results.empty:
        return pd.DataFrame()
    
    # Rename columns to match the names used in the original code
    classifier_results.columns = [
        'date', 'predicted_label', 'actual_label', 
        'prob_up', 'prob_side', 'prob_down'
    ]
    
    # Query market data for the same dates
    dates = tuple(classifier_results['date'].tolist())
    if len(dates) == 1:
        # Handle single date case with different SQL syntax
        market_query = select(
            MarketData.report_date, MarketData.open, MarketData.high, 
            MarketData.low, MarketData.close, MarketData.volume
        ).where(
            (MarketData.ticker == ticker) & 
            (MarketData.report_date == dates[0])
        ).order_by(MarketData.report_date)
    else:
        market_query = select(
            MarketData.report_date, MarketData.open, MarketData.high, 
            MarketData.low, MarketData.close, MarketData.volume
        ).where(
            (MarketData.ticker == ticker) & 
            (MarketData.report_date.in_(dates))
        ).order_by(MarketData.report_date)
    
    market_data = pd.DataFrame(db_session.execute(market_query).all())
    if not market_data.empty:
        market_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # Query indicators data
    indicator_query = select(
        EquityIndicators.report_date, 
        EquityIndicators.ema_20, 
        EquityIndicators.ema_50,
        EquityIndicators.ema_200
    ).where(
        (EquityIndicators.ticker == ticker) & 
        (EquityIndicators.report_date.in_(dates))
    ).order_by(EquityIndicators.report_date)
    
    indicators_data = pd.DataFrame(db_session.execute(indicator_query).all())
    if not indicators_data.empty:
        indicators_data.columns = ['date', 'ema_20', 'ema_50', 'ema_200']
    
    # Merge all data
    combined_data = classifier_results
    
    if not market_data.empty:
        combined_data = pd.merge(combined_data, market_data, on='date', how='left')
    
    if not indicators_data.empty:
        combined_data = pd.merge(combined_data, indicators_data, on='date', how='left')
    
    return combined_data

def calculate_actual_label_returns(prediction_data: pd.DataFrame, window_size: int = 20) -> Dict:
    """
    Calculate returns for each actual label type over the specified window.
    
    Parameters:
    - prediction_data: DataFrame with stock data and actual labels
    - window_size: The number of days to calculate returns over (default: 20)
    
    Returns:
    - Dictionary with return statistics for each label type
    """
    if prediction_data.empty:
        return {}
    
    # Initialize return lists for each label type
    returns_by_label = {0: [], 1: [], 2: []}  # 0=up, 1=side, 2=down
    
    # Ensure date column is a datetime
    prediction_data['date'] = pd.to_datetime(prediction_data['date'])
    
    # Calculate returns for each prediction
    for i in range(len(prediction_data) - window_size):
        current_price = prediction_data['close'].iloc[i]
        future_price = prediction_data['close'].iloc[i + window_size]
        
        # Calculate percentage return
        returns = (future_price - current_price) / current_price
        
        # Get the actual label for this point
        label = prediction_data['actual_label'].iloc[i]
        
        # Store the return with its corresponding label
        if label in returns_by_label:
            returns_by_label[label].append({
                'date': prediction_data['date'].iloc[i],
                'current_price': current_price,
                'future_price': future_price,
                'return': returns,
                'prediction': label,
                'prob_up': prediction_data['prob_up'].iloc[i],
                'prob_side': prediction_data['prob_side'].iloc[i],
                'prob_down': prediction_data['prob_down'].iloc[i]
            })
    
    # Convert to DataFrames for easier analysis
    dfs_by_label = {}
    for label in returns_by_label:
        if returns_by_label[label]:
            dfs_by_label[label] = pd.DataFrame(returns_by_label[label])
    
    # Calculate statistics for each label
    results = {}
    for label in dfs_by_label:
        df = dfs_by_label[label]
        results[label] = {
            'data': df,
            'count': len(df),
            'mean': df['return'].mean(),
            'std': df['return'].std(),
            'min': df['return'].min(),
            'max': df['return'].max(),
            'median': df['return'].median(),
            'positive_pct': (df['return'] > 0).mean() * 100  # % of positive returns
        }
    
    return results

def plot_combined_returns_distribution(returns_results: Dict, ticker: str) -> go.Figure:
    """Create a combined KDE plot of returns for all trend types using line curves."""
    # Create figure
    fig = go.Figure()
    
    # Define label names and colors
    label_names = {0: "Uptrend", 1: "Sideways", 2: "Downtrend"}
    colors = {0: "rgba(0, 128, 0, 0.8)", 1: "rgba(128, 128, 128, 0.8)", 2: "rgba(255, 0, 0, 0.8)"}
    fill_colors = {0: "rgba(0, 128, 0, 0.1)", 1: "rgba(128, 128, 128, 0.1)", 2: "rgba(255, 0, 0, 0.1)"}
    
    # Determine the overall min and max returns for proper x-axis scaling
    all_returns = []
    for label in returns_results:
        if label in returns_results and returns_results[label]['count'] > 0:
            all_returns.extend(returns_results[label]['data']['return'].tolist())
    
    if all_returns:
        min_return = min(all_returns)
        max_return = max(all_returns)
        # Calculate appropriate range for x-axis
        range_padding = (max_return - min_return) * 0.1
        x_min = min_return - range_padding
        x_max = max_return + range_padding
    else:
        x_min, x_max = -0.5, 0.5  # Default range if no data
    
    # Add distribution for each label type
    for label in sorted(returns_results.keys()):
        if label in returns_results and returns_results[label]['count'] > 0:
            # Get data for this label
            returns_df = returns_results[label]['data']
            count = returns_results[label]['count']
            
            # Create a kernel density estimate
            hist, bin_edges = np.histogram(returns_df['return'], bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Add line chart for the distribution
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=hist,
                mode='lines',
                name=f"{label_names[label]} (n={count})",
                line=dict(color=colors[label], width=3, shape='spline', smoothing=1.3),
                fill='tozeroy',
                fillcolor=fill_colors[label]
            ))
            
            # Add vertical line for the mean
            mean_return = returns_df['return'].mean()
            
            fig.add_vline(
                x=mean_return,
                line=dict(color=colors[label], width=2, dash='dash'),
            )
            
            # Position annotations with Uptrend highest, then Sideways, then Downtrend
            y_positions = {0: 0.95, 1: 0.85, 2: 0.75}  # Different vertical positions
            
            # Add annotation as a separate text element with background
            fig.add_annotation(
                x=mean_return,
                y=y_positions[label],
                text=f"{label_names[label]} Mean: {mean_return:.2%}",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=colors[label],
                ax=0,
                ay=30,
                font=dict(color="#000000", size=16),  # Increased font size for annotations
                align="center",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=colors[label],
                borderwidth=1,
                borderpad=3,
                xref="x", 
                yref="paper"
            )
    
    # Add vertical and horizontal lines at zero with solid black line
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    # Calculate adaptive tick values based on the data range
    range_span = x_max - x_min
    
    # Determine tick interval based on range span
    if range_span <= 0.2:  # Small range
        tick_interval = 0.05
    elif range_span <= 0.5:  # Medium range
        tick_interval = 0.1
    else:  # Large range
        tick_interval = 0.2
    
    # Generate tick values
    tick_values = []
    current = 0  # Always include zero
    
    # Add ticks to the left of zero
    while current >= x_min:
        tick_values.append(current)
        current -= tick_interval
    
    # Add ticks to the right of zero (skip zero as it's already added)
    current = tick_interval
    while current <= x_max:
        tick_values.append(current)
        current += tick_interval
    
    # Sort the tick values
    tick_values.sort()
    
    # Update layout
    fig.update_layout(
        title=f"Return Distribution by Trend Type for {ticker}",
        xaxis_title="Return",
        yaxis_title="Density",
        legend_title="Trend Type",
        xaxis=dict(
            tickformat='.0%',  # Shorter percentage format to avoid overlap
            tickmode='array',
            tickvals=tick_values,
            range=[x_min, x_max]
        ),
        hovermode="x unified",
        margin=dict(t=150, r=50, b=50, l=50, pad=10),  # Increased top margin and padding
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        font=dict(size=18),  # Increased font size
        width=1400,
        height=800,
        template="plotly_white"
    )
    
    return fig

def generate_return_distribution_plot(ticker: str, model: str, feature_set: str, window_size: int) -> None:
    """Generate and save return distribution plot as PNG file."""
    try:
        print(f"Generating return distribution plot for {ticker}...")
        
        # Create database connection
        engine = create_engine(DB_URL)
        DBSession = sessionmaker(bind=engine)
        
        with DBSession() as session:
            # Load data
            print(f"Loading prediction data for {ticker} with model {model} and feature set {feature_set}...")
            prediction_data = get_classifier_results(session, ticker, model, feature_set)
            
            if prediction_data.empty:
                print(f"No prediction data found for ticker {ticker}.")
                return
            
            # Calculate returns using actual labels
            print(f"Calculating returns with {window_size}-day window...")
            returns_results = calculate_actual_label_returns(prediction_data, window_size)
            
            if not returns_results:
                print("Could not calculate returns with the available data.")
                return
            
            # Create the combined plot
            print("Creating the combined return distribution plot...")
            combined_fig = plot_combined_returns_distribution(returns_results, ticker)
            
            # Define output directory and file name
            output_dir = os.path.join(os.path.dirname(__file__), "images")
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
            output_file = os.path.join(output_dir, f"{ticker}_combined_distribution_{window_size}days.png")
            
            # Save the figure as a PNG file
            print(f"Saving figure to {output_file}...")
            combined_fig.write_image(output_file, scale=2)  # scale=2 for higher resolution
            
            print(f"Successfully saved plot to {output_file}")
            
            # Print summary statistics for reference
            print("\nSummary Statistics:")
            print("=" * 50)
            label_names = {0: "Uptrend", 1: "Sideways", 2: "Downtrend"}
            
            for label in sorted(returns_results.keys()):
                if label in returns_results and returns_results[label]['count'] > 0:
                    stats = returns_results[label]
                    print(f"{label_names[label]} (n={stats['count']}):")
                    print(f"  Mean Return: {stats['mean']:.2%}")
                    print(f"  Median Return: {stats['median']:.2%}")
                    print(f"  Std Dev: {stats['std']:.2%}")
                    print(f"  % Positive Returns: {stats['positive_pct']:.1f}%")
                    print(f"  Min Return: {stats['min']:.2%}")
                    print(f"  Max Return: {stats['max']:.2%}")
                    print("-" * 40)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

def generate_combined_matrix_plot(model: str, feature_set: str, window_size: int) -> None:
    """Generate a single PNG file containing plots for all stocks in a 4-column and 7-row matrix."""
    try:
        print("Generating combined matrix plot for all stocks...")
        
        # Create database connection
        engine = create_engine(DB_URL)
        DBSession = sessionmaker(bind=engine)
        
        with DBSession() as session:
            # Get all available tickers
            tickers_query = select(ClassifierResult.ticker).distinct().order_by(ClassifierResult.ticker)
            tickers = [row[0] for row in session.execute(tickers_query).all()]
            
            if not tickers:
                print("No tickers found in the database.")
                return
            
            # Create a subplot figure with 7 rows and 4 columns
            rows_count = 7
            cols_count = 4
            
            fig = make_subplots(
                rows=rows_count, 
                cols=cols_count, 
                subplot_titles=tickers, 
                vertical_spacing=0.05,  # Reduced vertical spacing
                horizontal_spacing=0.05  # Reduced horizontal spacing
            )
            
            # Iterate over tickers and add plots
            for idx, ticker in enumerate(tickers):
                print(f"Processing {ticker} ({idx + 1}/{len(tickers)})...")
                
                # Load data for the ticker
                prediction_data = get_classifier_results(session, ticker, model, feature_set)
                
                if prediction_data.empty:
                    print(f"No prediction data found for ticker {ticker}. Skipping...")
                    continue
                
                # Calculate returns using actual labels
                returns_results = calculate_actual_label_returns(prediction_data, window_size)
                
                if not returns_results:
                    print(f"Could not calculate returns for ticker {ticker}. Skipping...")
                    continue
                
                # Generate the combined return distribution plot for the ticker
                ticker_fig = plot_combined_returns_distribution(returns_results, ticker)
                
                # Calculate the row and column for this subplot
                row = (idx // cols_count) + 1
                col = (idx % cols_count) + 1
                
                # Extract the data from the ticker's figure and add it to the subplot
                for trace in ticker_fig.data:
                    fig.add_trace(trace, row=row, col=col)
                
                # Determine the range for x-axis
                all_returns = []
                for label in returns_results:
                    if label in returns_results and returns_results[label]['count'] > 0:
                        all_returns.extend(returns_results[label]['data']['return'].tolist())
                
                if all_returns:
                    min_return = min(all_returns)
                    max_return = max(all_returns)
                    range_padding = (max_return - min_return) * 0.1
                    x_min = min_return - range_padding
                    x_max = max_return + range_padding
                    
                    # Add vertical line at zero
                    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, row=row, col=col)
                    
                    # Add vertical lines for the mean of each trend return
                    label_names = {0: "Uptrend", 1: "Sideways", 2: "Downtrend"}
                    colors = {0: "rgba(0, 128, 0, 0.8)", 1: "rgba(128, 128, 128, 0.8)", 2: "rgba(255, 0, 0, 0.8)"}
                    
                    # Add mean lines for each trend type
                    for label in sorted(returns_results.keys()):
                        if label in returns_results and returns_results[label]['count'] > 0:
                            mean_return = returns_results[label]['mean']
                            
                            # Add vertical line for the mean
                            fig.add_vline(
                                x=mean_return,
                                line_dash="dash",
                                line_color=colors[label],
                                line_width=2,
                                row=row,
                                col=col
                            )
                    
                    # Use the subplot axes directly for annotations
                    # This ensures correct positioning in each subplot
                    if row == 1 and col == 1:
                        xref = "x"
                        yref = "y"
                    else:
                        subplot_idx = (row - 1) * cols_count + col
                        xref = f"x{subplot_idx}"
                        yref = f"y{subplot_idx}"
                    
                    # Set the annotation positions as a fraction of the max data value
                    # Get the max y value from the data for this subplot
                    max_y_values = []
                    for label in returns_results:
                        if label in returns_results and returns_results[label]['count'] > 0:
                            hist, bin_edges = np.histogram(returns_results[label]['data']['return'], bins=50, density=True)
                            max_y_values.append(max(hist))
                    
                    if max_y_values:
                        max_y = max(max_y_values)
                        # Position the labels at the top using data coordinates
                        y_positions = {
                            0: max_y * 1.02, # Uptrend at 98% of max height
                            1: max_y * 0.9,# Sideways at 85% of max height
                            2: max_y * 0.78 # Downtrend at 72% of max height
                        }
                        
                        for label in sorted(returns_results.keys()):
                            if label in returns_results and returns_results[label]['count'] > 0:
                                mean_return = returns_results[label]['mean']
                                
                                # Add annotation using data coordinates
                                fig.add_annotation(
                                    x=x_max - range_padding * 0.2,  # Position near right edge
                                    y=y_positions[label],
                                    text=f"{label_names[label]}: {mean_return:.1%}",
                                    showarrow=False,
                                    font=dict(
                                        size=10,
                                        color=colors[label].replace("0.8", "1.0")
                                    ),
                                    xref=xref,
                                    yref=yref,
                                    align="right",
                                    xanchor="right",
                                    yanchor="top",
                                    bgcolor="rgba(255, 255, 255, 0.8)",
                                    bordercolor=colors[label],
                                    borderwidth=1,
                                    borderpad=3
                                )
                    
                    # Calculate adaptive tick values
                    range_span = x_max - x_min
                    if range_span <= 0.2:
                        tick_interval = 0.05
                    elif range_span <= 0.5:
                        tick_interval = 0.1
                    else:
                        tick_interval = 0.2
                    
                    # Generate tick values centered on zero
                    tick_values = [0]  # Always include zero
                    
                    # Add ticks to the left and right of zero
                    current = -tick_interval
                    while current >= x_min:
                        tick_values.append(current)
                        current -= tick_interval
                    
                    current = tick_interval
                    while current <= x_max:
                        tick_values.append(current)
                        current += tick_interval
                    
                    tick_values.sort()
                    
                    # Update axes for this subplot
                    fig.update_xaxes(
                        title_text="Return", 
                        row=row, 
                        col=col, 
                        tickformat=".0%",  # Shorter percentage format
                        title_font=dict(size=14),
                        tickfont=dict(size=12),
                        tickmode="array",
                        tickvals=tick_values,
                        range=[x_min, x_max]
                    )
            
            # Update layout for the combined figure
            fig.update_layout(
                showlegend=False,
                height=2100,  # Adjust height for better visibility
                width=1200,   # Adjust width for better visibility
                template="plotly_white",
                margin=dict(t=20, b=20, l=20, r=20),
                font=dict(size=16)  # Set font size for axis values and titles
            )
            
            # Define output directory and file name
            output_dir = os.path.join(os.path.dirname(__file__), "images")
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
            output_file = os.path.join(output_dir, f"combined_return_distribution_matrix.png")
            
            # Save the figure as a PNG file
            print(f"Saving combined matrix plot to {output_file}...")
            fig.write_image(output_file, scale=2)  # scale=2 for higher resolution
            
            print(f"Successfully saved combined matrix plot to {output_file}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate combined return distribution matrix plot for all stocks")
    parser.add_argument("--model", default="MLPv2", help="Model name (default: MLPv2)")
    parser.add_argument("--feature-set", default="processed technical indicators (20 days)", 
                        help="Feature set name (default: 'processed technical indicators (20 days)')")
    parser.add_argument("--window", type=int, default=20, help="Return window size in days (default: 20)")
    
    args = parser.parse_args()
    
    generate_combined_matrix_plot(args.model, args.feature_set, args.window)