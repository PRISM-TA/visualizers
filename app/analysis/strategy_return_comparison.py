# strategy_return_comparison.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to the path so we can import app modules
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
    page_title="Strategy Comparison",
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
def run_analysis(ticker, model, feature_set, strategy1, strategy2):
    """Run the analysis for both strategies and return the results"""
    session = get_db_connection()
    feeder = DataFeeder(session)
    trade_logger = TradeLogger(session)
    
    # Run analysis for first strategy (as target) vs second strategy (as benchmark)
    analyzer1 = CumRetAnalysis(feeder, trade_logger)
    analyzer1.setParam(
        CumRetAnalysisParam(
            benchmark_strategy=strategy2,
            benchmark_initial_capital=10000,
            target_strategy=strategy1,
            target_initial_capital=10000,
            classifier_model=model,
            feature_set=feature_set,
            ticker=ticker
        )
    )
    
    return analyzer1.run()

# Streamlit app
def main():
    # Title and description
    st.title("Strategy Comparison Visualizer")
    st.write("Compare cumulative returns of two trading strategies")
    
    # Sidebar for inputs
    st.sidebar.header("Comparison Parameters")
    
    # Define tickers
    all_tickers = ["AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", 
              "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", 
              "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "VZ", "WMT", "XOM"]
    
    # Define strategies
    strategies = ["RouletteStrategy", "BuyAndHoldStrategy", "MACDStrategy", "RSIStrategy"]
    
    # Input parameters
    ticker = st.sidebar.selectbox("Select Ticker", all_tickers)
    model = st.sidebar.text_input("Model", "CNNv0")
    feature_set = st.sidebar.text_input("Feature Set", "processed technical indicators (20 days)")
    strategy1 = st.sidebar.selectbox("Strategy 1", strategies)
    strategy2 = st.sidebar.selectbox("Strategy 2", strategies, index=1)  # Default to BuyAndHold as second strategy
    initial_investment = st.sidebar.number_input("Initial Investment ($)", 
                                                value=10000, 
                                                min_value=1, 
                                                step=1000)
    
    # Run analysis button
    if st.sidebar.button("Compare Strategies"):
        with st.spinner("Running comparison analysis..."):
            try:
                results_df = run_analysis(ticker, model, feature_set, strategy1, strategy2)
                
                # Determine column names based on strategies
                # Strategy 1 is target, Strategy 2 is benchmark
                strategy1_col = 'target_cum_return_pct'
                strategy2_col = 'benchmark_cum_return_pct'
                strategy1_daily_col = 'target_return_change_pct'
                strategy2_daily_col = 'benchmark_return_change_pct'
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Cumulative Returns", "Performance Difference", "Daily Returns", "Data Table"])
                
                with tab1:
                    # Main content - Cumulative Return Comparison
                    st.header(f"Cumulative Return Comparison: {ticker}")
                    
                    # Calculate final values
                    strategy1_final_value = initial_investment * (1 + results_df[strategy1_col].iloc[-1]/100)
                    strategy2_final_value = initial_investment * (1 + results_df[strategy2_col].iloc[-1]/100)
                    
                    # Create the cumulative return plot with Plotly
                    fig = go.Figure()
                    
                    # Add strategy1 cumulative return line
                    fig.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=results_df[strategy1_col],
                        mode='lines',
                        name=strategy1,
                        line=dict(color='#1f77b4', width=2.5)
                    ))
                    
                    # Add strategy2 cumulative return line
                    fig.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=results_df[strategy2_col],
                        mode='lines',
                        name=strategy2,
                        line=dict(color='#ff7f0e', width=2.5)
                    ))
                    
                    # Add zero line
                    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black", opacity=0.5)
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Cumulative Return Comparison: {strategy1} vs {strategy2} - {ticker}",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return (%)",
                        height=600,
                        hovermode="x unified",
                        legend_title="Strategies",
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    
                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display key statistics in metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(strategy1)
                        subcol1, subcol2, subcol3 = st.columns(3)
                        subcol1.metric("Initial Investment", f"${initial_investment:,.2f}")
                        subcol2.metric("Final Value", f"${strategy1_final_value:,.2f}")
                        subcol3.metric("Total Return", f"{results_df[strategy1_col].iloc[-1]:.2f}%")
                    
                    with col2:
                        st.subheader(strategy2)
                        subcol1, subcol2, subcol3 = st.columns(3)
                        subcol1.metric("Initial Investment", f"${initial_investment:,.2f}")
                        subcol2.metric("Final Value", f"${strategy2_final_value:,.2f}")
                        subcol3.metric("Total Return", f"{results_df[strategy2_col].iloc[-1]:.2f}%")
                
                with tab2:
                    # Performance Difference
                    st.header(f"Performance Difference: {strategy1} vs {strategy2}")
                    
                    # Calculate performance difference (how much strategy1 outperforms strategy2)
                    performance_diff = results_df['target_performance_pct']  # This is already calculated in the analyzer
                    
                    # Create performance difference plot
                    fig_perf = go.Figure()
                    
                    # Add performance line
                    fig_perf.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=performance_diff,
                        mode='lines',
                        name=f"{strategy1} vs {strategy2}",
                        line=dict(color='green', width=2)
                    ))
                    
                    # Add zero line
                    fig_perf.add_hline(y=0, line_width=1, line_dash="solid", line_color="black", opacity=0.5)
                    
                    # Add shaded areas for positive and negative performance
                    fig_perf.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=performance_diff.where(performance_diff > 0, 0),
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 0, 0.2)',
                        line=dict(width=0),
                        name='Outperformance'
                    ))
                    
                    fig_perf.add_trace(go.Scatter(
                        x=results_df['date'],
                        y=performance_diff.where(performance_diff <= 0, 0),
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(width=0),
                        name='Underperformance'
                    ))
                    
                    # Update layout
                    fig_perf.update_layout(
                        title=f"Performance Difference: {strategy1} vs {strategy2} - {ticker}",
                        xaxis_title="Date",
                        yaxis_title="Performance Difference (%)",
                        height=500,
                        hovermode="x unified"
                    )
                    
                    # Display the figure
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    # Add some statistics
                    final_outperformance = performance_diff.iloc[-1]
                    avg_outperformance = performance_diff.mean()
                    positive_days = (performance_diff > 0).sum()
                    positive_pct = positive_days / len(performance_diff) * 100
                    
                    # Display statistics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Final Outperformance", f"{final_outperformance:.2f}%", 
                               delta=f"{final_outperformance:.2f}%", delta_color="normal")
                    col2.metric("Average Outperformance", f"{avg_outperformance:.3f}%")
                    col3.metric("Days Outperforming", f"{positive_days}")
                    col4.metric("Outperformance %", f"{positive_pct:.1f}%")
                
                with tab3:
                    # Daily Returns Comparison
                    st.header(f"Daily Returns Comparison: {ticker}")
                    
                    # Create daily returns plot
                    fig_daily = go.Figure()
                    
                    # Add strategy1 daily returns
                    fig_daily.add_trace(go.Bar(
                        x=results_df['date'],
                        y=results_df[strategy1_daily_col],
                        name=strategy1,
                        marker_color='rgba(31, 119, 180, 0.7)'
                    ))
                    
                    # Add strategy2 daily returns
                    fig_daily.add_trace(go.Bar(
                        x=results_df['date'],
                        y=results_df[strategy2_daily_col],
                        name=strategy2,
                        marker_color='rgba(255, 127, 14, 0.7)'
                    ))
                    
                    # Update layout
                    fig_daily.update_layout(
                        title=f"Daily Returns: {strategy1} vs {strategy2} - {ticker}",
                        xaxis_title="Date",
                        yaxis_title="Daily Return (%)",
                        height=500,
                        barmode='group',
                        hovermode="x unified"
                    )
                    
                    # Display the figure
                    st.plotly_chart(fig_daily, use_container_width=True)
                    
                    # Statistics for daily returns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"{strategy1} Daily Stats")
                        st1_avg = results_df[strategy1_daily_col].mean()
                        st1_std = results_df[strategy1_daily_col].std()
                        st1_pos = (results_df[strategy1_daily_col] > 0).sum()
                        st1_pos_pct = (st1_pos / len(results_df)) * 100
                        
                        subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                        subcol1.metric("Avg Daily Return", f"{st1_avg:.3f}%")
                        subcol2.metric("Daily Std Dev", f"{st1_std:.3f}%")
                        subcol3.metric("Positive Days", f"{st1_pos}")
                        subcol4.metric("Win Rate", f"{st1_pos_pct:.1f}%")
                    
                    with col2:
                        st.subheader(f"{strategy2} Daily Stats")
                        st2_avg = results_df[strategy2_daily_col].mean()
                        st2_std = results_df[strategy2_daily_col].std()
                        st2_pos = (results_df[strategy2_daily_col] > 0).sum()
                        st2_pos_pct = (st2_pos / len(results_df)) * 100
                        
                        subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                        subcol1.metric("Avg Daily Return", f"{st2_avg:.3f}%")
                        subcol2.metric("Daily Std Dev", f"{st2_std:.3f}%")
                        subcol3.metric("Positive Days", f"{st2_pos}")
                        subcol4.metric("Win Rate", f"{st2_pos_pct:.1f}%")
                
                with tab4:
                    # Raw data table
                    st.header("Raw Data")
                    
                    # Select relevant columns for comparison
                    compare_cols = ['date', 
                                    strategy1_col, strategy1_daily_col, 
                                    strategy2_col, strategy2_daily_col,
                                    'target_performance_pct']
                    
                    # Create a more readable dataframe
                    display_df = results_df[compare_cols].copy()
                    
                    # Rename columns for clarity
                    column_map = {
                        strategy1_col: f'{strategy1} Cum Return (%)',
                        strategy1_daily_col: f'{strategy1} Daily Return (%)',
                        strategy2_col: f'{strategy2} Cum Return (%)',
                        strategy2_daily_col: f'{strategy2} Daily Return (%)',
                        'target_performance_pct': 'Performance Difference (%)',
                        'date': 'Date'
                    }
                    display_df = display_df.rename(columns=column_map)
                    
                    # Show the data
                    st.dataframe(display_df)
                    
                    # Add download button for CSV
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_{strategy1}_vs_{strategy2}_comparison.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error running analysis: {e}")
                st.exception(e)
    else:
        st.info("Select parameters and click 'Compare Strategies' to view the comparison")
        
        # Show sample chart when no data is loaded
        placeholder_fig = go.Figure()
        
        # Sample data for strategy 1
        x_dates = [datetime(2023, 1, 1) + pd.Timedelta(days=x) for x in range(100)]
        y_strategy1 = np.cumsum(np.random.normal(0.001, 0.02, 100)) * 100
        y_strategy2 = np.cumsum(np.random.normal(0.0005, 0.01, 100)) * 100
        
        # Add sample traces
        placeholder_fig.add_trace(go.Scatter(
            x=x_dates,
            y=y_strategy1,
            mode='lines',
            name='Strategy 1 (Sample)',
            line=dict(color='#1f77b4', width=2)
        ))
        
        placeholder_fig.add_trace(go.Scatter(
            x=x_dates,
            y=y_strategy2,
            mode='lines',
            name='Strategy 2 (Sample)',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        placeholder_fig.update_layout(
            title="Sample Comparison (Select parameters and click 'Compare Strategies' for real data)",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400,
            legend_title="Strategies"
        )
        st.plotly_chart(placeholder_fig, use_container_width=True)

if __name__ == "__main__":
    main()