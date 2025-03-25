from app.models.TradeLog import TradeLog
import pandas as pd

def calculate_pnl(initial_capital: float, trade_logs: list[TradeLog]) -> float:
    current_capital = initial_capital
    holdings = {}  # Dictionary to track holdings per ticker

    for trade in trade_logs:
        if trade.action == 'BUY':
            current_capital -= trade.price * trade.shares

            if trade.ticker in holdings:
                holdings[trade.ticker] += trade.shares
            else:
                holdings[trade.ticker] = trade.shares

        elif trade.action == 'SELL':
            if trade.ticker in holdings:
                sell_amount = trade.shares * trade.price
                holdings[trade.ticker] -= trade.shares
                current_capital += sell_amount
            else:
                print(f"No sufficient holdings found for {trade.ticker} to sell")

        else:
            print(f"Unexpected action type: {trade.action}")
    
    pnl = current_capital - initial_capital
    return pnl

def calculate_returns(initial_capital: float, trade_logs: list[TradeLog]) -> pd.DataFrame:
    """
    Calculate cumulative returns and daily changes from trade logs and return as DataFrame.
    
    Args:
        initial_capital: Initial investment amount
        trade_logs: List of trade log entries
    
    Returns:
        DataFrame with columns: 'cumulative_return', 'daily_change'
    """
    # Sort trade logs by date
    sorted_trades = sorted(trade_logs, key=lambda x: x.report_date)
    
    current_capital = initial_capital
    holdings = {}  # Dictionary to track holdings per ticker
    data = []  # List to store daily data
    previous_return = 0
    
    for trade in sorted_trades:
        date_key = trade.report_date
        
        # Process trade
        if trade.action == 'BUY':
            current_capital -= trade.price * trade.shares
            
            if trade.ticker in holdings:
                holdings[trade.ticker] += trade.shares
            else:
                holdings[trade.ticker] = trade.shares
                
        elif trade.action == 'SELL':
            if trade.ticker in holdings:
                sell_amount = trade.shares * trade.price
                holdings[trade.ticker] -= trade.shares
                current_capital += sell_amount
            else:
                print(f"No sufficient holdings found for {trade.ticker} to sell")
        
        # Calculate returns
        cumulative_return = ((current_capital - initial_capital) / initial_capital) * 100
        daily_change = cumulative_return - previous_return
        
        # Store daily data
        data.append({
            'date': date_key,
            'cumulative_return': cumulative_return,
            'daily_change': daily_change,
            'capital': current_capital
        })
        
        previous_return = cumulative_return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Remove duplicate dates, keeping the last entry for each date
    df = df[~df.index.duplicated(keep='last')]
    
    return df
