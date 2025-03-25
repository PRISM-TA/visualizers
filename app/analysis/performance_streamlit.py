# performance_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to the path so we can import app modules
# Adjust this path to point to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Now import the app modules
try:
    from app.db.session import create_db_session
    from app.analysis.CumRetAnalysis import CumRetAnalysis, CumRetAnalysisParam
    from app.datafeed.DataFeeder import DataFeeder
    from app.datafeed.TradeLogger import TradeLogger
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.info(f"Current project root is set to: {project_root}")
    st.info("Please adjust the 'project_root' path in the script to point to your project root directory.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Strategy Performance Analysis",
    page_icon="📈",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Database connection
@st.cache_resource
def get_db_connection():
    return create_db_session(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        port=None if os.getenv("DB_PORT") == "None" else os.getenv("DB_PORT")
    )

# Data processing
@st.cache_data
def run_analysis(ticker, model, feature_set, target_strategy, benchmark_strategy):
    """Run the analysis and return the results"""
    session = get_db_connection()
    feeder = DataFeeder(session)
    trade_logger = TradeLogger(session)
    
    analyzer = CumRetAnalysis(feeder, trade_logger)
    analyzer.setParam(
        CumRetAnalysisParam(
            benchmark_strategy=benchmark_strategy,
            benchmark_initial_capital=10000,
            target_strategy=target_strategy,
            target_initial_capital=10000,
            classifier_model=model,
            feature_set=feature_set,
            ticker=ticker
        )
    )
    
    return analyzer.run()

# Streamlit app
def main():
    # Title and description
    st.title("Strategy Performance Analysis")
    st.write("Compare target strategy performance against benchmark")
    
    # Sidebar for inputs
    st.sidebar.header("Analysis Parameters")
    
    # Define tickers
    all_tickers = ["AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", 
              "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", 
              "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "VZ", "WMT", "XOM"]
    
    # Input parameters
    ticker = st.sidebar.selectbox("Select Ticker", all_tickers)
    model = st.sidebar.text_input("Model", "CNNv0")
    feature_set = st.sidebar.text_input("Feature Set", "processed technical indicators (20 days)")
    target_strategy = st.sidebar.text_input("Target Strategy", "RouletteStrategy")
    benchmark_strategy = st.sidebar.text_input("Benchmark Strategy", "BuyAndHoldStrategy")
    
    # Run analysis button
    if st.sidebar.button("Run Analysis"):
        with st.spinner("Running analysis..."):
            try:
                results_df = run_analysis(ticker, model, feature_set, target_strategy, benchmark_strategy)
                
                # Main content
                st.header(f"Performance Analysis: {ticker}")
                
                # Create tabs for different visualizations
                tab1, tab2 = st.tabs(["Performance Plot", "Data Table"])
                
                with tab1:
                    # Create the performance plot with Plotly
                    fig = go.Figure()
                    
                    # Add performance line
                    fig.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=results_df['target_performance_pct'],
                        mode='lines',
                        name='Performance Difference',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add zero line
                    fig.add_hline(y=0, line_width=2, line_dash="solid", line_color="black", opacity=0.5)
                    
                    # Add shaded areas for positive and negative performance
                    fig.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=results_df['target_performance_pct'].where(results_df['target_performance_pct'] > 0, 0),
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 0, 0.2)',
                        line=dict(width=0),
                        name='Positive Performance'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=results_df['target_performance_pct'].where(results_df['target_performance_pct'] <= 0, 0),
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(width=0),
                        name='Negative Performance'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Strategy Performance: {target_strategy} vs {benchmark_strategy} - {ticker}",
                        xaxis_title="Date",
                        yaxis_title="Performance Difference (%)",
                        legend_title="Legend",
                        height=600,
                        hovermode="x unified"
                    )
                    
                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add some statistics
                    final_outperformance = results_df['target_performance_pct'].iloc[-1]
                    avg_outperformance = results_df['target_performance_pct'].mean()
                    positive_days = (results_df['target_performance_pct'] > 0).sum()
                    positive_pct = positive_days / len(results_df) * 100
                    
                    # Display statistics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Final Outperformance", f"{final_outperformance:.2f}%")
                    col2.metric("Average Daily Outperformance", f"{avg_outperformance:.3f}%")
                    col3.metric("Positive Days", f"{positive_days}")
                    col4.metric("Positive Percentage", f"{positive_pct:.1f}%")
                    
                with tab2:
                    # Show the raw data
                    st.dataframe(results_df)
                    
                    # Add download button for CSV
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_performance_analysis.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error running analysis: {e}")
                st.exception(e)
    else:
        st.info("Select parameters and click 'Run Analysis' to view results")

if __name__ == "__main__":
    main()