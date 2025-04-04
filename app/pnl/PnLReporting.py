from app.models.TradeLog import TradeLog

# TODO: use shares instead of sharess
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
       