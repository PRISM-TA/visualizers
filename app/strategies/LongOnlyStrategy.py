"""
Long-Only Strategy Description

Overview:
This is a trend-following strategy that takes long-only positions based on machine learning predictions of market conditions. The strategy uses a classifier model that predicts whether the market is in an uptrend, sideways, or downtrend state.

Entry Conditions:
- Enter long positions only when the classifier predicts an uptrend
- No short positions are taken (hence "Long-Only")
- New positions are only taken when not currently holding a position

Exit Conditions:
The strategy exits positions under three scenarios:
1. Stop Loss: Exit if the position loses more than the specified stop loss percentage (default -5%)
2. Counter Threshold: Exit if the classifier shows non-uptrend predictions for a consecutive number of days (default 3 days)
3. Holding Period: The uptrend prediction validity is limited to a specified number of days (default 20 days)

Parameters:
- sell_counter_threshold: Number of consecutive non-uptrend days before forced exit (default 3 days)
- stop_loss_percentage: Maximum allowed loss before position is closed (default -5%)
- holding_period: Maximum number of days to hold based on a single uptrend prediction (default 20 days)

Trade Management:
- Each trade is recorded with entry/exit dates, prices, and reasons for exit
- Position sizing is fixed (100% of capital)
- Returns are compounded across trades
- Performance is tracked through trade recordsx

Risk Management:
- Stop loss orders to limit downside risk
- No leverage used
- Single position at a time
- Automatic exit on sustained trend reversal signals
"""
from app.models.MarketCondition import MarketCondition
from app.models.TradeLog import TradeLog

from app.strategies.BaseStrategy import BaseStrategy, BaseStrategyParam
from app.datafeed.DataFeeder import DataFeeder

from datetime import date


class LongOnlyStrategyParam(BaseStrategyParam):
    def __init__(
        self,
        sell_counter_threshold: int = 3,
        stop_loss_percentage: float = -0.05,
        holding_period: int = 20,
        initial_capital: float = 10000
    ):
        super().__init__(initial_capital)
        self.sell_counter_threshold = sell_counter_threshold
        self.stop_loss_percentage = stop_loss_percentage
        self.holding_period = holding_period

class LongOnlyStrategy(BaseStrategy):
    strategy_name: str = "LongOnlyStrategy"

    buy_spot: float
    sell_spot: float
    shares: float
    bought: bool
    day_counter: int
    non_buy_counter: int
    trades: list[TradeLog] = []

    def __init__(self, datafeeder: DataFeeder, param: LongOnlyStrategyParam):
        super().__init__(datafeeder, param)
        
        # Trading state
        self.shares = 0
        self.buy_spot = 0
        self.sell_spot = 0
        self.bought = False
        self.day_counter = 0
        self.non_buy_counter = 0

    def _handle_buy(self, ticker: str, date: date, price: float, shares: float):
        self.buy_spot = price
        self.bought = True
        self.day_counter = 0
        self.non_buy_counter = 0
        super()._handle_buy(ticker, date, price, shares)

    def _handle_sell(self, ticker: str, date: date, price: float, shares: float, reason: str):
        self.sell_spot = price
        self.bought = False
        self.day_counter = 0
        self.non_buy_counter = 0
        super()._handle_sell(ticker, date, price, shares, reason)

    def run(self, ticker: str, model: str, feature_set: str):
        datafeed = self.datafeeder.pullData(ticker=ticker, classifier_model=model, feature_set=feature_set)
        current_capital = self.param.initial_capital

        for daily_data in datafeed:
            # Buy MarketCondition
            if not self.bought and daily_data.predicted_label == MarketCondition.uptrend:
                self.shares = current_capital / daily_data.close
                self._handle_buy(
                    ticker, 
                    daily_data.report_date, 
                    daily_data.close, 
                    self.shares
                )
                
                current_capital -= daily_data.close * self.shares
                continue

            # Update counters
            if daily_data.predicted_label == MarketCondition.uptrend:
                self.non_buy_counter = 0
                self.day_counter += 1
                if self.day_counter == self.param.holding_period:
                    self.non_buy_counter = self.param.sell_counter_threshold - 1
            else:
                self.non_buy_counter += 1

            # Check stop loss
            if self.bought:
                current_loss_percentage = (daily_data.close - self.buy_spot) / self.buy_spot
                if current_loss_percentage <= self.param.stop_loss_percentage:
                    self._handle_sell(
                        ticker,
                        daily_data.report_date, 
                        daily_data.close,
                        self.shares,
                        f"Stop loss triggered at {current_loss_percentage:.2%}"
                    )

                    current_capital += daily_data.close * self.shares
                    self.shares = 0
                    continue

            # Check sell counter threshold
            if self.bought and self.non_buy_counter >= self.param.sell_counter_threshold:
                self._handle_sell(
                    ticker,
                    daily_data.report_date,
                    daily_data.close,
                    self.shares,
                    "Sell counter threshold reached"
                )

                current_capital += daily_data.close * self.shares
                self.shares = 0
                continue
        
        if self.bought:
            self._handle_sell(ticker, daily_data.report_date, daily_data.close, self.shares, "End of trading period")

    def dump_trade_logs(self) -> list[TradeLog]:
        return self.trades