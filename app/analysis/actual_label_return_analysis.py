# actual_labels_return_analysis.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
import math
from typing import Dict, List, Tuple, Any
import sys
from datetime import datetime
import io
from plotly.io import write_image

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from models.ClassifierResult import ClassifierResult
from models.MarketData import MarketData
from models.EquityIndicators import EquityIndicators
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Construct database URL 
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

st.set_page_config(
    page_title="Actual Labels Returns Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database helper functions using SQLAlchemy ORM
def get_available_tickers(db_session) -> List[str]:
    """Get list of all tickers that have classifier results."""
    query = select(ClassifierResult.ticker).distinct().order_by(ClassifierResult.ticker)
    results = db_session.execute(query).scalars().all()
    return results

def get_available_models(db_session, ticker: str) -> List[str]:
    """Get list of all models for a specific ticker."""
    query = select(ClassifierResult.model).distinct().where(
        ClassifierResult.ticker == ticker
    ).order_by(ClassifierResult.model)
    results = db_session.execute(query).scalars().all()
    return results

def get_available_feature_sets(db_session, ticker: str, model: str) -> List[str]:
    """Get list of all feature sets for a specific ticker and model."""
    query = select(ClassifierResult.feature_set).distinct().where(
        (ClassifierResult.ticker == ticker) & (ClassifierResult.model == model)
    ).order_by(ClassifierResult.feature_set)
    results = db_session.execute(query).scalars().all()
    return results

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
        
        # Get the actual label for this point (using actual_label instead of predicted_label)
        label = prediction_data['actual_label'].iloc[i]
        
        # Store the return with its corresponding label
        if label in returns_by_label:
            returns_by_label[label].append({
                'date': prediction_data['date'].iloc[i],
                'current_price': current_price,
                'future_price': future_price,
                'return': returns,
                'prediction': label,  # Use actual_label but keep the column name as 'prediction' for consistency
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
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
    
    # Update layout
    fig.update_layout(
        title=f"Return Distribution by Trend Type for {ticker}",
        xaxis_title="Return",
        yaxis_title="Density",
        legend_title="Trend Type",
        xaxis_tickformat='.1%',
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
    )
    
    return fig

def plot_returns_histogram(returns_data: pd.DataFrame, label: int, label_name: str, ticker: str) -> go.Figure:
    """Create a histogram of returns for a given actual label."""
    # Choose color based on label
    color = 'green' if label == 0 else 'gray' if label == 1 else 'red'
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=returns_data['return'],
        nbinsx=30,
        marker_color=color,
        opacity=0.75,
        name="Returns"
    ))
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    
    # Add mean line
    mean_return = returns_data['return'].mean()
    fig.add_vline(
        x=mean_return, 
        line_dash="solid", 
        line_color="blue", 
        annotation=dict(
            text=f"Mean: {mean_return:.2%}",
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=0
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"Distribution of Returns for '{label_name}' Trends ({ticker})",
        xaxis_title="Return",
        yaxis_title="Count",
        showlegend=False,
        xaxis_tickformat='.1%',
        margin=dict(t=120, r=50, b=50, l=50, pad=10),  # Increased margins
        font=dict(size=16),  # Increased font size
    )
    
    return fig

def plot_returns_over_time(returns_data: pd.DataFrame, label: int, label_name: str, ticker: str) -> go.Figure:
    """Create a line chart of returns over time for a given actual label."""
    # Sort data by date
    data = returns_data.sort_values('date')
    
    # Choose color based on label
    color = 'green' if label == 0 else 'gray' if label == 1 else 'red'
    
    # Create figure
    fig = go.Figure()
    
    # Add line chart
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['return'],
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(color=color, size=4),
        name="Returns"
    ))
    
    # Add horizontal line at zero return
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    # Update layout
    fig.update_layout(
        title=f"Returns Over Time for '{label_name}' Trends ({ticker})",
        xaxis_title="Date",
        yaxis_title="Return",
        showlegend=False,
        yaxis_tickformat='.1%',
        margin=dict(t=120, r=50, b=50, l=50, pad=10),  # Increased margins
        font=dict(size=16),  # Increased font size
    )
    
    return fig

def plot_returns_vs_probability(returns_data: pd.DataFrame, label: int, label_name: str, ticker: str) -> go.Figure:
    """Create a scatter plot of returns vs. prediction probability."""
    # Choose probability column based on label
    prob_column = 'prob_up' if label == 0 else 'prob_side' if label == 1 else 'prob_down'
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=returns_data[prob_column],
        y=returns_data['return'],
        mode='markers',
        marker=dict(
            color=returns_data['return'],
            colorscale='RdYlGn',
            cmin=min(returns_data['return'].min(), -0.05),  # Set lower bound to at least -5%
            cmax=max(returns_data['return'].max(), 0.05),   # Set upper bound to at least +5%
            colorbar=dict(title='Return'),
            size=10,
            opacity=0.7
        ),
        name="Returns"
    ))
    
    # Add horizontal line at zero return
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    # Update layout
    fig.update_layout(
        title=f"Returns vs. Prediction Probability for '{label_name}' Trends ({ticker})",
        xaxis_title=f"{label_name} Probability",
        yaxis_title="Return",
        showlegend=False,
        yaxis_tickformat='.1%',
        xaxis_range=[0, 1],
        margin=dict(t=120, r=50, b=50, l=50, pad=10),  # Increased margins
        font=dict(size=16),  # Increased font size
    )
    
    return fig

def plot_cumulative_returns(returns_data: pd.DataFrame, label: int, label_name: str, ticker: str) -> go.Figure:
    """Create a chart showing cumulative returns if following trends of this type."""
    # Sort by date
    sorted_data = returns_data.sort_values('date').copy()
    
    # Calculate cumulative returns (1 + r1) * (1 + r2) * ... - 1
    sorted_data['cum_return'] = (1 + sorted_data['return']).cumprod() - 1
    
    # Choose color based on label
    color = 'green' if label == 0 else 'gray' if label == 1 else 'red'
    
    # Create figure
    fig = go.Figure()
    
    # Add line chart
    fig.add_trace(go.Scatter(
        x=sorted_data['date'],
        y=sorted_data['cum_return'],
        mode='lines',
        line=dict(color=color, width=2),
        name="Cumulative Returns"
    ))
    
    # Add horizontal line at zero
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    # Update layout
    fig.update_layout(
        title=f"Cumulative Returns Following '{label_name}' Trends ({ticker})",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        showlegend=False,
        yaxis_tickformat='.1%',
        margin=dict(t=120, r=50, b=50, l=50, pad=10),  # Increased margins
        font=dict(size=16),  # Increased font size
    )
    
    return fig

def download_plot_as_image(fig: go.Figure, filename: str) -> None:
    """
    Provide a download button for a Plotly chart as an image in Streamlit.

    Parameters:
    - fig: Plotly Figure object to export.
    - filename: Name of the file to download (e.g., 'chart.png').
    """
    # Create a BytesIO buffer to store the image
    buffer = io.BytesIO()
    
    # Write the figure to the buffer as a PNG image
    write_image(fig, buffer, format="png", width=1200, height=800)
    
    # Reset the buffer's position to the beginning
    buffer.seek(0)
    
    # Provide a download button in Streamlit
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="image/png"
    )

def main():
    try:
        # Initialize session state
        if 'window_size' not in st.session_state:
            st.session_state.window_size = 20
        
        # Create database connection
        engine = create_engine(DB_URL)
        DBSession = sessionmaker(bind=engine)
        
        with DBSession() as session:
            # Page title
            st.title("Actual Labels Returns Analysis")
            st.write("Analyze the returns achieved by following actual trend labels")
            
            # Sidebar for selection options
            with st.sidebar:
                st.header("Data Selection")
                
                # Load available tickers
                tickers = get_available_tickers(session)
                if not tickers:
                    st.error("No ticker data found in the database.")
                    return
                
                ticker = st.selectbox("Select Ticker:", tickers)
                
                # Hardcode model and feature set
                model = "MLPv2"
                feature_set = "processed technical indicators (20 days)"
                
                st.header("Analysis Settings")
                window_size = st.slider(
                    "Return Window Size (days)", 
                    min_value=1, 
                    max_value=60, 
                    value=st.session_state.window_size,
                    help="Number of days to calculate returns over"
                )
                st.session_state.window_size = window_size
                
                st.header("Legend")
                st.markdown("🟢 Up (0): Actual uptrend")
                st.markdown("⚫ Side (1): Actual sideways movement")
                st.markdown("🔴 Down (2): Actual downtrend")
                
                # Display the fixed model and feature set
                # st.header("Model Information")
                # st.info(f"Using model: {model}")
                # st.info(f"Feature set: {feature_set}")
            
            # Load data
            with st.spinner("Loading prediction data..."):
                prediction_data = get_classifier_results(session, ticker, model, feature_set)
                
                if prediction_data.empty:
                    st.error(f"No prediction data found for ticker {ticker} with model {model} and feature set {feature_set}.")
                    return
            
            # Calculate returns using actual labels
            with st.spinner("Calculating returns..."):
                returns_results = calculate_actual_label_returns(prediction_data, window_size)
                
                if not returns_results:
                    st.error("Could not calculate returns with the available data.")
                    return
            
            # Display overall statistics
            st.header(f"Returns Summary ({window_size}-day Window)")
            
            # Create a summary table
            label_names = {0: "Up", 1: "Side", 2: "Down"}
            summary_data = []
            
            for label in sorted(returns_results.keys()):
                if label in returns_results and returns_results[label]['count'] > 0:
                    stats = returns_results[label]
                    summary_data.append({
                        "Trend": label_names[label],
                        "Count": stats['count'],
                        "Mean Return": f"{stats['mean']:.2%}",
                        "Median Return": f"{stats['median']:.2%}",
                        "Std Dev": f"{stats['std']:.2%}",
                        "Min Return": f"{stats['min']:.2%}",
                        "Max Return": f"{stats['max']:.2%}",
                        "% Positive Returns": f"{stats['positive_pct']:.1f}%"
                    })
            
            # Display summary table
            if summary_data:
                st.table(pd.DataFrame(summary_data))
                
                # Add combined distribution chart
                st.header("Combined Return Distribution")
                combined_fig = plot_combined_returns_distribution(returns_results, ticker)
                st.plotly_chart(combined_fig, use_container_width=True)
                download_plot_as_image(combined_fig, f"{ticker}_combined_distribution.png")
            
            # Detailed analysis by prediction type
            st.header("Detailed Analysis by Trend Type")
            
            tab_labels = []
            for label in sorted(returns_results.keys()):
                if label in returns_results and returns_results[label]['count'] > 0:
                    tab_labels.append(label)
            
            if tab_labels:
                tabs = st.tabs([label_names[label] for label in tab_labels])
                
                for i, label in enumerate(tab_labels):
                    with tabs[i]:
                        stats = returns_results[label]
                        returns_df = stats['data']
                        
                        st.write(f"Analysis of returns when the trend was '{label_names[label]}'")
                        
                        # Display key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Return", f"{stats['mean']:.2%}")
                        with col2:
                            st.metric("Median Return", f"{stats['median']:.2%}")
                        with col3:
                            st.metric("Standard Deviation", f"{stats['std']:.2%}")
                        with col4:
                            st.metric("% Positive Returns", f"{stats['positive_pct']:.1f}%")
                        
                        # Create visualizations
                        st.subheader("Return Distribution")
                        hist_fig = plot_returns_histogram(returns_df, label, label_names[label], ticker)
                        st.plotly_chart(hist_fig, use_container_width=True)
                        download_plot_as_image(hist_fig, f"{ticker}_{label_names[label]}_distribution.png")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Returns Over Time")
                            time_fig = plot_returns_over_time(returns_df, label, label_names[label], ticker)
                            st.plotly_chart(time_fig, use_container_width=True)
                            download_plot_as_image(time_fig, f"{ticker}_{label_names[label]}_returns_over_time.png")
                        
                        with col2:
                            st.subheader("Cumulative Returns")
                            cum_fig = plot_cumulative_returns(returns_df, label, label_names[label], ticker)
                            st.plotly_chart(cum_fig, use_container_width=True)
                            download_plot_as_image(cum_fig, f"{ticker}_{label_names[label]}_cumulative_returns.png")
                        
                        st.subheader("Returns vs. Prediction Probability")
                        prob_fig = plot_returns_vs_probability(returns_df, label, label_names[label], ticker)
                        st.plotly_chart(prob_fig, use_container_width=True)
                        download_plot_as_image(prob_fig, f"{ticker}_{label_names[label]}_returns_vs_probability.png")
                        
                        # Show raw data
                        with st.expander("View Raw Data"):
                            st.dataframe(
                                returns_df.sort_values('date')[
                                    ['date', 'current_price', 'future_price', 'return', 
                                     'prob_up', 'prob_side', 'prob_down']
                                ].reset_index(drop=True)
                            )
            else:
                st.warning("No return data available for analysis.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.exception(traceback.format_exc())

if __name__ == "__main__":
    main()