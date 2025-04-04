from app.models.MarketCondition import MarketCondition
from app.models.TradeLog import TradeLog

from app.strategies.BaseStrategy import BaseStrategy, BaseStrategyParam
from app.datafeed.DataFeeder import DataFeeder

from datetime import date


class ShortOnlyStrategyParam(BaseStrategyParam):
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

class ShortOnlyStrategy(BaseStrategy):
    strategy_name: str = "ShortOnlyStrategy"

    buy_spot: float
    sell_spot: float
    shares: float
    bought: bool
    day_counter: int
    non_buy_counter: int
    trades: list[TradeLog] = []

    def __init__(self, datafeeder: DataFeeder, param: ShortOnlyStrategyParam):
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
            if not self.bought and daily_data.predicted_label == MarketCondition.downtrend:
                self.shares = -1 * current_capital / daily_data.close
                self._handle_buy(
                    ticker, 
                    daily_data.report_date, 
                    daily_data.close, 
                    self.shares
                )
                
                current_capital -= daily_data.close * self.shares
                continue

            # Update counters
            if daily_data.predicted_label == MarketCondition.downtrend:
                self.non_buy_counter = 0
                self.day_counter += 1
                if self.day_counter == self.param.holding_period:
                    self.non_buy_counter = self.param.sell_counter_threshold - 1
            else:
                self.non_buy_counter += 1

            # Check stop loss
            if self.bought:
                current_loss_percentage = -1 * (daily_data.close - self.buy_spot) / self.buy_spot
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