import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from app.db.session import create_db_session
from app.strategies.BAHStrategy import BAHStrategy, BAHParam
from app.strategies.RouletteStrategy import RouletteStrategy, RouletteStrategyParam, DecisionFactory
from app.strategies.LongOnlyStrategy import LongOnlyStrategy, LongOnlyStrategyParam
from app.strategies.ShortOnlyStrategy import ShortOnlyStrategy, ShortOnlyStrategyParam
from app.pnl.PnLReporting import calculate_pnl
from app.datafeed.DataFeeder import DataFeeder
from dotenv import load_dotenv

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
def get_db_session():
    return create_db_session(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        port=os.getenv("DB_PORT")
    )

# Get DataFeeder
@st.cache_resource
def get_data_feeder():
    session = get_db_session()
    return DataFeeder(session)

# Available strategies
STRATEGIES = [
    "BuyAndHoldStrategy",
    "RouletteStrategy",
    "LongOnlyStrategy",
    "ShortOnlyStrategy"
]

# Default tickers
TICKERS = [
    "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", 
    "GE", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", 
    "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", 
    "UTX", "VZ", "WMT", "XOM"
]

# Create strategy instance
def create_strategy(strategy_name, initial_capital=10000):
    feeder = get_data_feeder()
    
    if strategy_name == "BuyAndHoldStrategy":
        params = BAHParam(initial_capital=initial_capital)
        return BAHStrategy(feeder, params)
    
    elif strategy_name == "RouletteStrategy":
        params = RouletteStrategyParam(
            initial_capital=initial_capital,
            roulette_size=20,
            decision_factory=DecisionFactory
        )
        return RouletteStrategy(feeder, params)
    
    elif strategy_name == "LongOnlyStrategy":
        params = LongOnlyStrategyParam(
            initial_capital=initial_capital,
            sell_counter_threshold=3,
            stop_loss_percentage=-0.05,
            holding_period=20
        )
        return LongOnlyStrategy(feeder, params)
    
    elif strategy_name == "ShortOnlyStrategy":
        params = ShortOnlyStrategyParam(
            initial_capital=initial_capital,
            sell_counter_threshold=3,
            stop_loss_percentage=-0.05,
            holding_period=20
        )
        return ShortOnlyStrategy(feeder, params)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

# Run strategy and get PnL
@st.cache_data(ttl=300)
def run_strategy(strategy_name, ticker, model, feature_set, initial_capital):
    try:
        # Create and reset strategy
        strategy = create_strategy(strategy_name, initial_capital)
        strategy.reset()
        
        # Run strategy
        strategy.run(ticker=ticker, model=model, feature_set=feature_set)
        
        # Get trade logs
        trade_logs = strategy.dump_trade_logs()
        
        # Calculate PnL
        pnl = calculate_pnl(initial_capital, trade_logs)
        
        # Generate daily returns data for visualization
        daily_data = generate_daily_data(strategy_name, initial_capital, trade_logs)
        
        return {
            'strategy_name': strategy_name,
            'pnl': pnl,
            'final_value': initial_capital + pnl,
            'return_pct': (pnl / initial_capital) * 100,
            'trade_logs': trade_logs,
            'daily_data': daily_data
        }
    except Exception as e:
        st.error(f"Error running {strategy_name}: {str(e)}")
        st.exception(e)
        return None

# Generate daily data for visualization
def generate_daily_data(strategy_name, initial_capital, trade_logs):
    """Generate daily data points for strategy visualization"""
    if not trade_logs:
        return pd.DataFrame()
    
    # Sort trade logs by date
    sorted_trades = sorted(trade_logs, key=lambda x: x.report_date)
    
    # Track portfolio state
    cash = initial_capital
    holdings = {}  # ticker -> shares
    
    # For storing daily data
    data = []
    prev_date = None
    
    # Process each trade
    for trade in sorted_trades:
        date = trade.report_date.strftime('%Y-%m-%d')
        
        # Handle trade
        if trade.action == 'BUY':
            cost = trade.price * trade.shares
            cash -= cost
            holdings[trade.ticker] = holdings.get(trade.ticker, 0) + trade.shares
            
        elif trade.action == 'SELL':
            proceeds = trade.price * trade.shares
            cash += proceeds
            if trade.ticker in holdings:
                holdings[trade.ticker] = max(0, holdings.get(trade.ticker, 0) - trade.shares)
        
        # Only add data point if date changed (to get end-of-day values)
        if date != prev_date:
            # Calculate holdings value
            holdings_value = 0
            for ticker, shares in holdings.items():
                if shares > 0:
                    # Find most recent price for this ticker
                    latest_price = next((t.price for t in reversed(sorted_trades) 
                                        if t.ticker == ticker and t.report_date <= trade.report_date), 0)
                    holdings_value += shares * latest_price
            
            # Calculate total value and return
            total_value = cash + holdings_value
            pnl = total_value - initial_capital
            return_pct = (pnl / initial_capital) * 100
            
            # Store data point
            data.append({
                'date': pd.to_datetime(date),
                'cash': cash,
                'holdings_value': holdings_value,
                'total_value': total_value,
                'pnl': pnl,
                'return_pct': return_pct
            })
            
            prev_date = date
    
    # Create dataframe
    return pd.DataFrame(data)

# Run comparison of two strategies
@st.cache_data(ttl=300)
def compare_strategies(ticker, model, feature_set, strategy1_name, strategy2_name, initial_capital):
    """Compare two strategies and return results for visualization"""
    # Run both strategies
    result1 = run_strategy(strategy1_name, ticker, model, feature_set, initial_capital)
    result2 = run_strategy(strategy2_name, ticker, model, feature_set, initial_capital)
    
    if not result1 or not result2:
        return None
    
    # Get daily data for each strategy
    daily1 = result1['daily_data']
    daily2 = result2['daily_data']
    
    # Check if we have data to visualize
    if daily1.empty and daily2.empty:
        return {
            'strategy1': result1,
            'strategy2': result2,
            'comparison_data': pd.DataFrame()
        }
    
    # Create dummy data if one strategy has no trades
    if daily1.empty:
        start_date = daily2['date'].min()
        end_date = daily2['date'].max()
        daily1 = pd.DataFrame({
            'date': [start_date, end_date],
            'cash': [initial_capital, initial_capital],
            'holdings_value': [0, 0],
            'total_value': [initial_capital, initial_capital],
            'pnl': [0, 0],
            'return_pct': [0, 0]
        })
    
    if daily2.empty:
        start_date = daily1['date'].min()
        end_date = daily1['date'].max()
        daily2 = pd.DataFrame({
            'date': [start_date, end_date],
            'cash': [initial_capital, initial_capital],
            'holdings_value': [0, 0],
            'total_value': [initial_capital, initial_capital],
            'pnl': [0, 0],
            'return_pct': [0, 0]
        })
    
    # Rename columns for clarity
    daily1 = daily1.rename(columns={
        'cash': f'cash_{strategy1_name}',
        'holdings_value': f'holdings_value_{strategy1_name}',
        'total_value': f'total_value_{strategy1_name}',
        'pnl': f'pnl_{strategy1_name}',
        'return_pct': f'return_pct_{strategy1_name}'
    })
    
    daily2 = daily2.rename(columns={
        'cash': f'cash_{strategy2_name}',
        'holdings_value': f'holdings_value_{strategy2_name}',
        'total_value': f'total_value_{strategy2_name}',
        'pnl': f'pnl_{strategy2_name}',
        'return_pct': f'return_pct_{strategy2_name}'
    })
    
    # Get all dates from both strategies
    all_dates = pd.DataFrame({
        'date': pd.date_range(
            start=min(daily1['date'].min(), daily2['date'].min()),
            end=max(daily1['date'].max(), daily2['date'].max()),
            freq='D'
        )
    })
    
    # Merge data
    merged1 = pd.merge(all_dates, daily1, on='date', how='left')
    merged2 = pd.merge(all_dates, daily2, on='date', how='left')
    
    # Forward fill missing values
    for df in [merged1, merged2]:
        for col in df.columns:
            if col != 'date':
                df[col] = df[col].fillna(method='ffill')
    
    # Final merge
    comparison_data = pd.merge(merged1, merged2, on='date', how='outer')
    
    # Add difference column
    comparison_data[f'diff_{strategy1_name}_vs_{strategy2_name}'] = (
        comparison_data[f'return_pct_{strategy1_name}'] - 
        comparison_data[f'return_pct_{strategy2_name}']
    )
    
    return {
        'strategy1': result1,
        'strategy2': result2,
        'comparison_data': comparison_data
    }

# UI Layout
st.title("Trading Strategy Comparison")

# Sidebar inputs
st.sidebar.header("Configuration")

# Model and feature set
model = st.sidebar.selectbox("Model", ["MLPv2"], index=0)
feature_set = st.sidebar.selectbox(
    "Feature Set", 
    ["processed technical indicators (20 days)"],
    index=0
)

# Strategy selection
strategy1 = st.sidebar.selectbox("Strategy 1", STRATEGIES, index=0)
strategy2 = st.sidebar.selectbox("Strategy 2", STRATEGIES, index=1)

# Ticker selection
ticker = st.sidebar.selectbox("Ticker", TICKERS, index=0)

# Initial capital
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

# Run comparison button
if st.sidebar.button("Run Comparison"):
    with st.spinner("Running strategy comparison..."):
        results = compare_strategies(ticker, model, feature_set, strategy1, strategy2, initial_capital)
        
        if results:
            st.session_state.results = results
        else:
            st.error("Failed to run comparison. See error messages above.")

# Display results
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    
    # Summary metrics
    st.header("Performance Summary")
    
    col1, col2 = st.columns(2)
    
    # Strategy 1 metrics
    with col1:
        st.subheader(f"{strategy1}")
        st.metric(
            "Final PnL", 
            f"${results['strategy1']['pnl']:,.2f}",
            f"{results['strategy1']['return_pct']:,.2f}%"
        )
    
    # Strategy 2 metrics
    with col2:
        st.subheader(f"{strategy2}")
        st.metric(
            "Final PnL", 
            f"${results['strategy2']['pnl']:,.2f}",
            f"{results['strategy2']['return_pct']:,.2f}%"
        )
    
    # Comparison chart
    st.header("Return Comparison")

    if not results['comparison_data'].empty:
        fig = go.Figure()
        
        # Add return lines for both strategies - no difference line
        fig.add_trace(go.Scatter(
            x=results['comparison_data']['date'],
            y=results['comparison_data'][f'return_pct_{strategy1}'],
            mode='lines',
            name=f"{strategy1}",
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=results['comparison_data']['date'],
            y=results['comparison_data'][f'return_pct_{strategy2}'],
            mode='lines',
            name=f"{strategy2}",
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{ticker}: Strategy Return Comparison",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            legend=dict(x=0, y=1),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for visualization.")
    
    # Debug information
    with st.expander("Debug Information"):
        st.subheader("Raw PnL Values")
        st.write(f"{strategy1}: ${results['strategy1']['pnl']:,.2f}")
        st.write(f"{strategy2}: ${results['strategy2']['pnl']:,.2f}")
        
        st.subheader("First 5 Rows of Comparison Data")
        if not results['comparison_data'].empty:
            st.dataframe(results['comparison_data'].head())
        
        st.subheader("Raw Trade Logs Count")
        st.write(f"{strategy1}: {len(results['strategy1']['trade_logs'])}")
        st.write(f"{strategy2}: {len(results['strategy2']['trade_logs'])}")
else:
    st.info("Select parameters and click 'Run Comparison' to start.")