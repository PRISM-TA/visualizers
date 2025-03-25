import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from datetime import datetime
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from sqlalchemy import select, create_engine, text
from sqlalchemy.orm import sessionmaker
from models.ClassifierResult import ClassifierResult
from models.MarketData import MarketData
from models.EquityIndicators import EquityIndicators
from dotenv import load_dotenv
from typing import List, Dict, Tuple

st.set_page_config(layout="wide")
st.title("Model Behavior Visualizer")

# Initialize session state for current index if it doesn't exist
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# Load environment variables from .env
load_dotenv()

# Construct database URL as in analyze_accuracy.py
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

def get_available_tickers(db_session) -> List[str]:
    """Get list of all tickers that have classifier results."""
    query = select(ClassifierResult.ticker).distinct().order_by(ClassifierResult.ticker)
    results = db_session.execute(query).scalars().all()
    return results

def get_available_models(db_session, ticker: str) -> List[str]:
    """Get list of all models used for a specific ticker."""
    query = (
        select(ClassifierResult.model)
        .where(ClassifierResult.ticker == ticker)
        .distinct()
        .order_by(ClassifierResult.model)
    )
    results = db_session.execute(query).scalars().all()
    return results

def get_available_feature_sets(db_session, ticker: str, model: str) -> List[str]:
    """Get list of all feature sets used for a specific ticker and model."""
    query = (
        select(ClassifierResult.feature_set)
        .where(ClassifierResult.ticker == ticker)
        .where(ClassifierResult.model == model)
        .distinct()
        .order_by(ClassifierResult.feature_set)
    )
    results = db_session.execute(query).scalars().all()
    return results

def get_classifier_results(db_session, ticker: str, model: str, feature_set: str) -> pd.DataFrame:
    """Get classifier results for specific ticker, model, and feature set."""
    query = (
        select(
            ClassifierResult.report_date,
            ClassifierResult.actual_label,
            ClassifierResult.predicted_label,
            ClassifierResult.downtrend_prob,
            ClassifierResult.side_prob,
            ClassifierResult.uptrend_prob
        )
        .where(ClassifierResult.ticker == ticker)
        .where(ClassifierResult.model == model)
        .where(ClassifierResult.feature_set == feature_set)
        .order_by(ClassifierResult.report_date)
    )
    
    results = db_session.execute(query).all()
    
    if not results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=[
        'date', 'actual_label', 'predicted_label', 
        'prob_down', 'prob_side', 'prob_up'
    ])
    
    # Convert dates to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def get_market_data(db_session, ticker: str, start_date=None, end_date=None) -> pd.DataFrame:
    """Get market data with indicators for a specific ticker."""
    query = (
        select(
            MarketData.report_date, 
            MarketData.open, 
            MarketData.high, 
            MarketData.low, 
            MarketData.close, 
            MarketData.volume,
            EquityIndicators.ema_20,
            EquityIndicators.ema_50,
            EquityIndicators.ema_200
        )
        .join(
            EquityIndicators,
            (MarketData.ticker == EquityIndicators.ticker) &
            (MarketData.report_date == EquityIndicators.report_date)
        )
        .where(MarketData.ticker == ticker)
    )
    
    if start_date:
        query = query.where(MarketData.report_date >= start_date)
    if end_date:
        query = query.where(MarketData.report_date <= end_date)
        
    query = query.order_by(MarketData.report_date)
    
    results = db_session.execute(query).all()
    
    if not results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=[
        'date', 'open', 'high', 'low', 'close', 'volume',
        'ema_20', 'ema_50', 'ema_200'
    ])
    
    # Convert dates to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def merge_data(market_data: pd.DataFrame, prediction_data: pd.DataFrame) -> pd.DataFrame:
    """Merge market data with prediction data based on date."""
    if market_data.empty or prediction_data.empty:
        return pd.DataFrame()
    
    # Merge on date
    merged_df = pd.merge(market_data, prediction_data, on='date', how='inner')
    
    return merged_df

def export_plotly_to_png(fig, filename=None):
    """
    Export a Plotly figure as PNG image and offer download
    """
    if filename is None:
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_export_{timestamp}.png"
    
    # Create a deep copy of the figure to avoid modifying the original
    export_fig = go.Figure(fig)
    
    # Adjust layout for better spacing in exported PNG
    export_fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=125, b=80),  # Increase top margin to prevent title overlap
        title=dict(
            y=0.95,  # Move title position up
            x=0.5,
            xanchor='center',
            yanchor='top'
        )
    )
    
    # Create a BytesIO object to store the image
    img_bytes = BytesIO()
    
    # Write the figure to the BytesIO object as PNG
    export_fig.write_image(img_bytes, format='png', engine='kaleido', width=1200, height=900)
    
    # Reset the pointer to the start of the BytesIO object
    img_bytes.seek(0)
    
    # Encode the bytes as base64 for download
    b64_png = base64.b64encode(img_bytes.read()).decode()
    
    # Create the HTML download link
    href = f'<a download="{filename}" href="data:image/png;base64,{b64_png}">Download Plot as PNG</a>'
    
    return href


def plot_model_behavior(combined_data: pd.DataFrame, ticker: str, model: str, feature_set: str, 
                     start_idx: int, window_size: int = 100, display_mode: str = "prediction_type") -> go.Figure:
    """
    Create a plot showing price and model behavior.
    
    Parameters:
    - combined_data: DataFrame with price and prediction data
    - ticker: Stock symbol
    - model: Model name
    - feature_set: Feature set name
    - start_idx: Starting index for the window
    - window_size: Size of the data window to display
    - display_mode: Either "prediction_type", "prediction_accuracy", or "actual_labels_only"
    """
    if combined_data.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig
    
     # Get the window of data to display
    if start_idx >= len(combined_data):
        start_idx = 0
    
    end_idx = min(start_idx + window_size, len(combined_data))
    window_df = combined_data.iloc[start_idx:end_idx].copy()
    
    # Get date range for title
    date_range = "N/A"
    if not window_df.empty:
        start_date = window_df['date'].iloc[0].strftime('%Y-%m-%d')
        end_date = window_df['date'].iloc[-1].strftime('%Y-%m-%d')
        date_range = f"{start_date} to {end_date}"
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.2, 0.3],
        vertical_spacing=0.08,
        shared_xaxes=True,
        subplot_titles=(
            "Price Chart with Predicted Labels (Labels stands for the trend of the next 20 days)", 
            "Volume", 
            "Class Probabilities"
        )
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=window_df['date'],
            open=window_df['open'],
            high=window_df['high'],
            low=window_df['low'],
            close=window_df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add EMAs to the price subplot
    ema_colors = {'ema_20': '#FF4500', 'ema_50': '#9370DB', 'ema_200': '#CD853F'}
    for ema, color in ema_colors.items():
        if ema in window_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=window_df['date'],
                    y=window_df[ema],
                    name=ema.upper(),
                    line=dict(color=color, width=1)
                ),
                row=1, col=1
            )
    
    # Add volume
    colors = ['red' if row['close'] < row['open'] else 'green' 
             for _, row in window_df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=window_df['date'],
            y=window_df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.3
        ),
        row=2, col=1
    )
    
    # Add class probabilities first (they don't change between display modes)
    fig.add_trace(
        go.Scatter(
            x=window_df['date'],
            y=window_df['prob_up'],
            mode='lines',
            name='Up Probability',
            line=dict(color='green', width=1)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=window_df['date'],
            y=window_df['prob_side'],
            mode='lines',
            name='Side Probability',
            line=dict(color='gray', width=1)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=window_df['date'],
            y=window_df['prob_down'],
            mode='lines',
            name='Down Probability',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    # Add a reference line at 0.5 for probabilities
    fig.add_shape(
        type="line",
        line=dict(dash="dash", width=1, color="black"),
        x0=window_df['date'].min(),
        x1=window_df['date'].max(),
        y0=0.5,
        y1=0.5,
        row=3, col=1
    )
    
    if display_mode == "prediction_type":
        # DISPLAY MODE 1: Show prediction types (Up/Side/Down)
        label_colors = {0: 'green', 1: 'gray', 2: 'red'}
        label_names = {0: 'Up', 1: 'Side', 2: 'Down'}
        
        for label in [0, 1, 2]:
            label_data = window_df[window_df['predicted_label'] == label]
            if not label_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=label_data['date'],
                        y=label_data['close'],
                        mode='markers',
                        name=f'Predicted: {label_names[label]}',
                        marker=dict(
                            color=label_colors[label],
                            size=8,
                            symbol='circle'
                        ),
                        hovertemplate=(
                            "Date: %{x}<br>"
                            "Close: %{y}<br>"
                            "Prediction: " + label_names[label] + "<br>"
                            "Down Prob: %{customdata[0]:.3f}<br>"
                            "Side Prob: %{customdata[1]:.3f}<br>"
                            "Up Prob: %{customdata[2]:.3f}"
                        ),
                        customdata=label_data[['prob_down', 'prob_side', 'prob_up']]
                    ),
                    row=1, col=1
                )
        
        # Update subplot title for clarity
        fig.layout.annotations[0].text = "Price Chart with Predicted Trend Direction (Up/Side/Down)"
        
    elif display_mode == "prediction_accuracy":
        # DISPLAY MODE 2: Show prediction accuracy (Correct/Wrong)
        
        # Determine if predictions are correct by comparing with actual label
        if 'actual_label' in window_df.columns:
            window_df['is_correct'] = window_df['predicted_label'] == window_df['actual_label']
        else:
            # Fallback if actual_label is not available (for testing only)
            # In a real implementation, you should use actual ground truth
            window_df['is_correct'] = True  # Default to all correct for demonstration
        
        correct_data = window_df[window_df['is_correct'] == True]
        wrong_data = window_df[window_df['is_correct'] == False]
        
        # Show correct predictions
        if not correct_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=correct_data['date'],
                    y=correct_data['close'],
                    mode='markers',
                    name='Correct Prediction',
                    marker=dict(
                        color='green',
                        size=8,
                        symbol='circle'
                    ),
                    hovertemplate=(
                        "Date: %{x}<br>"
                        "Close: %{y}<br>"
                        "Prediction: Correct<br>"
                        "Predicted Label: %{customdata[0]}<br>"
                        "Actual Label: %{customdata[1]}<br>"
                        "Down Prob: %{customdata[2]:.3f}<br>"
                        "Side Prob: %{customdata[3]:.3f}<br>"
                        "Up Prob: %{customdata[4]:.3f}"
                    ),
                    customdata=correct_data[['predicted_label', 'actual_label', 
                                           'prob_down', 'prob_side', 'prob_up']]
                ),
                row=1, col=1
            )
        
        # Show wrong predictions
        if not wrong_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=wrong_data['date'],
                    y=wrong_data['close'],
                    mode='markers',
                    name='Wrong Prediction',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='x'
                    ),
                    hovertemplate=(
                        "Date: %{x}<br>"
                        "Close: %{y}<br>"
                        "Prediction: Wrong<br>"
                        "Predicted Label: %{customdata[0]}<br>"
                        "Actual Label: %{customdata[1]}<br>"
                        "Down Prob: %{customdata[2]:.3f}<br>"
                        "Side Prob: %{customdata[3]:.3f}<br>"
                        "Up Prob: %{customdata[4]:.3f}"
                    ),
                    customdata=wrong_data[['predicted_label', 'actual_label', 
                                         'prob_down', 'prob_side', 'prob_up']]
                ),
                row=1, col=1
            )
        
        # Update subplot title for clarity
        fig.layout.annotations[0].text = "Price Chart with Prediction Accuracy (Correct/Wrong)"
    
    elif display_mode == "actual_labels_only":
        # DISPLAY MODE 3: Show only actual labels
        label_colors = {0: 'green', 1: 'gray', 2: 'red'}
        label_names = {0: 'Up', 1: 'Side', 2: 'Down'}
        
        if 'actual_label' in window_df.columns:
            for label in [0, 1, 2]:
                label_data = window_df[window_df['actual_label'] == label]
                if not label_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=label_data['date'],
                            y=label_data['close'],
                            mode='markers',
                            name=f'Actual: {label_names[label]}',
                            marker=dict(
                                color=label_colors[label],
                                size=10,
                                symbol='diamond',  # Different symbol to distinguish from predictions
                                line=dict(width=1, color='black')  # Add border for visibility
                            ),
                            hovertemplate=(
                                "Date: %{x}<br>"
                                "Close: %{y}<br>"
                                "Actual Label: " + label_names[label] + "<br>"
                            )
                        ),
                        row=1, col=1
                    )
            
            # Update subplot title for clarity
            fig.layout.annotations[0].text = "Price Chart with Actual Trend Direction (Ground Truth)"
        else:
            # Show a message if actual labels are not available
            fig.add_annotation(
                text="Actual labels not available in dataset",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=18, color="red")
            )
            fig.layout.annotations[0].text = "Price Chart (Actual Labels Not Available)"
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Prediction Analysis ({model} model with {feature_set} features)',
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(t=150, l=80, r=80, b=90),
        legend=dict(
            orientation="h",  # Make legend horizontal
            yanchor="bottom",
            y=1.03,          # Position legend above the plot
            xanchor="center",
            x=0.5
        )
    )
    
    # Update y-axis range for probabilities
    fig.update_yaxes(range=[0, 1], title_text="Probability", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Update x-axis date format
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]) # hide weekends
        ],
        row=1, col=1
    )
    
    # Add date range to title
    fig.update_layout(
        title=f"{ticker} - {model} ({feature_set}) - {date_range}",
    )
    
    return fig

def display_metrics(data: pd.DataFrame):
    """Display prediction metrics and distribution."""
    if data.empty or 'predicted_label' not in data.columns:
        st.warning("No prediction data available for metrics calculation")
        return
    
    st.subheader("Prediction Distribution")
    
    # Label distribution
    label_counts = data['predicted_label'].value_counts().sort_index()
    label_percentage = (label_counts / label_counts.sum() * 100).round(2)
    label_names = {0: "Down", 1: "Side", 2: "Up"}
    
    # Create a dataframe for display
    stats_df = pd.DataFrame({
        'Label': [label_names.get(i, i) for i in label_counts.index],
        'Count': label_counts.values,
        'Percentage (%)': label_percentage.values
    })
    
    # Display label distribution
    cols = st.columns(3)
    for i, (_, row) in enumerate(stats_df.iterrows()):
        with cols[i % 3]:
            st.metric(f"{row['Label']} Predictions", 
                    f"{row['Count']} ({row['Percentage (%)']:.2f}%)")
    
    # If actual labels exist, display performance metrics
    if 'actual_label' in data.columns and not data['actual_label'].isna().all():
        st.subheader("Performance Metrics")
        
        # Calculate overall accuracy
        accuracy = (data['predicted_label'] == data['actual_label']).mean()
        
        # Calculate confusion matrix
        labels = sorted(pd.concat([
            data['actual_label'].dropna(), 
            data['predicted_label'].dropna()
        ]).unique())
        
        conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                conf_matrix[i, j] = len(
                    data[(data['actual_label'] == true_label) & 
                         (data['predicted_label'] == pred_label)]
                )
        
        # Calculate class-specific metrics
        metrics = []
        
        for i, cls in enumerate(labels):
            # True positives, false positives, false negatives
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.append({
                'Class': label_names.get(cls, cls),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1 Score': round(f1, 4)
            })
        
        # Display overall accuracy
        st.metric("Overall Accuracy", f"{accuracy:.2%}")
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        conf_df = pd.DataFrame(
            conf_matrix,
            index=[f'True_{label_names.get(l, l)}' for l in labels],
            columns=[f'Pred_{label_names.get(l, l)}' for l in labels]
        )
        st.dataframe(conf_df)
        
        # Display class-specific metrics
        st.subheader("Class-Specific Metrics")
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df)

def main():
    try:
        # Initialize session state for current_idx if it doesn't exist
        if 'current_idx' not in st.session_state:
            st.session_state.current_idx = 0
        if 'current_fig' not in st.session_state:
            st.session_state.current_fig = None
        
        # Create database connection as in analyze_accuracy.py
        engine = create_engine(DB_URL)
        DBSession = sessionmaker(bind=engine)
        
        with DBSession() as session:
            # Sidebar for selection options
            with st.sidebar:
                st.header("Model Selection")
                
                # Load available tickers
                tickers = get_available_tickers(session)
                if not tickers:
                    st.error("No ticker data found in the database.")
                    return
                
                ticker = st.selectbox("Select Ticker:", tickers)
                
                # Load available models for the selected ticker
                models = get_available_models(session, ticker)
                if not models:
                    st.error(f"No models found for ticker {ticker}")
                    return
                
                model = st.selectbox("Select Model:", models)
                
                # Load available feature sets for the selected ticker and model
                feature_sets = get_available_feature_sets(session, ticker, model)
                if not feature_sets:
                    st.error(f"No feature sets found for ticker {ticker} and model {model}")
                    return
                
                feature_set = st.selectbox("Select Feature Set:", feature_sets)
                
                st.header("Display Options")
                window_size = st.slider("Window Size (days)", min_value=20, max_value=250, value=100)
                
                # NEW: Display mode selector
                st.header("Label Display Mode")
                display_mode = st.radio(
                    "Choose what to display on price chart:", 
                    ["Prediction Type (Up/Side/Down)", "Prediction Accuracy (Correct/Wrong)", "actual_labels_only"],
                    index=0
                )
                # Convert selected option to parameter value
                if "Type" in display_mode:
                    display_param = "prediction_type"
                elif "Accuracy" in display_mode:
                    display_param = "prediction_accuracy"
                elif "actual_labels_only" in display_mode:
                    display_param = "actual_labels_only"
                
                # Show appropriate legend based on display mode
                st.header("Legend")
                if display_param == "prediction_type":
                    st.markdown("🔴 Down (2)")
                    st.markdown("⚫ Side (1)")
                    st.markdown("🟢 Up (0)")
                else:
                    st.markdown("🟢 Correct Prediction")
                    st.markdown("❌ Wrong Prediction")
            
            # Load data
            with st.spinner("Loading model predictions..."):
                prediction_data = get_classifier_results(session, ticker, model, feature_set)
                
                if prediction_data.empty:
                    st.error("No prediction data found for the selected parameters.")
                    return
                
                # Get start and end dates from prediction data
                min_date = prediction_data['date'].min()
                max_date = prediction_data['date'].max()
            
            with st.spinner("Loading market data..."):
                # Get market data for the ticker within the prediction date range
                market_data = get_market_data(session, ticker, min_date, max_date)
                
                if market_data.empty:
                    st.error("No market data found for the selected ticker.")
                    return
                
                # Merge market data with prediction data
                combined_data = merge_data(market_data, prediction_data)
                
                if combined_data.empty:
                    st.error("Failed to merge market data with prediction data.")
                    return
            
            # Navigation controls
            col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

            with col1:
                if st.button("⏮️ Start"):
                    st.session_state.current_idx = 0

            with col2:
                if st.button("⬅️ Previous") and st.session_state.current_idx >= window_size:
                    st.session_state.current_idx -= window_size

            with col3:
                st.write(f"Window {st.session_state.current_idx // window_size + 1} of {math.ceil(len(combined_data) / window_size)}")

            with col4:
                if st.button("Next ➡️") and st.session_state.current_idx + window_size < len(combined_data):
                    st.session_state.current_idx += window_size

            # Create and display plot 
            fig = plot_model_behavior(
                combined_data, ticker, model, feature_set, 
                st.session_state.current_idx, window_size, display_param
            )
            st.session_state.current_fig = fig  # Store in session state for export

            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            with st.spinner("Calculating metrics..."):
                display_metrics(combined_data)
            
            # Add option to download the data
            csv = combined_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data",
                data=csv,
                file_name=f"{ticker}_{model}_{feature_set}_data.csv",
                mime="text/csv",
            )
            
            # Add export to PNG button
            if st.session_state.current_fig is not None:
                export_filename = f"{ticker}_{model}_{feature_set}_chart.png"
                export_link = export_plotly_to_png(st.session_state.current_fig, export_filename)
                st.markdown(export_link, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()